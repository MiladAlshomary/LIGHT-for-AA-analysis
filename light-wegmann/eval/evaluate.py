import argparse
import csv
import json
import logging
import math
import numpy as np
import os
import shutil
import sys
import tempfile
import time
import torch
torch.set_float32_matmul_precision('high')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from collections import defaultdict
from multiprocessing import Process, Queue
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in tqdm(f, desc=f"Loading {file_path}")]

def extract_texts(dataset_name, entry):
    texts = entry.get("syms")
    if not isinstance(texts, list):
        raise TypeError(f"Expected a list in 'syms'. Got: {type(texts)}. Keys: {list(entry.keys())}")
    return texts

def load_texts_and_authors(dataset_name, file_path):
    data = load_jsonl(file_path)
    all_texts, all_authors = [], []
    for entry in tqdm(data, desc=f"Parsing {file_path}"):
        texts = extract_texts(dataset_name, entry)
        all_texts.extend(texts)
        all_authors.extend([entry["author_id"]] * len(texts))
    return all_texts, all_authors

def encode_multi_gpu(texts, authors, batch_size, prefix_path, model_str):
    """
    Encodes texts using all GPUs via SentenceTransformer's multi-process pool.
    Saves the result to disk at the specified prefix path. Returns embeddings and authors.
    """
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if len(devices) == 0:
        raise RuntimeError("No CUDA devices visible!")
    per_gpu_bs = max(1, math.ceil(batch_size / len(devices)))
    logging.info(
        f"Spawning {len(devices)} workers "
        f"(1 per GPU). Global batch={batch_size} → per‑GPU batch={per_gpu_bs}"
    )
    model = SentenceTransformer(
        model_str,
        device=devices[0]
    )
    model = model.to(dtype=torch.bfloat16)
    pool = model.start_multi_process_pool(target_devices=devices)
    embeddings_np = model.encode_multi_process(
        texts,
        pool,
        batch_size=per_gpu_bs,
        chunk_size=per_gpu_bs,
        show_progress_bar=True
    )

    model.stop_multi_process_pool(pool)

    # Convert to tensor manually
    embeddings = torch.from_numpy(embeddings_np)
    print(f"Embedding dtype: {embeddings.dtype}")

    final_path = f"{prefix_path}_all.pt"
    obj = {
        "embeddings": embeddings.cpu(),
        "authors": authors
    }

    logging.info(f"Writing embeddings to {os.path.abspath(final_path)}")
    torch.save(obj, final_path)

    return embeddings, authors

def group_and_average(embeddings, authors):
    author_to_embs = defaultdict(list)
    for emb, author in zip(embeddings, authors):
        author_to_embs[author].append(emb)
    averaged_embeddings, author_ids = [], []
    for author, emb_list in author_to_embs.items():
        avg = torch.stack(emb_list).mean(dim=0)
        averaged_embeddings.append(avg)
        author_ids.append(author)
    return torch.stack(averaged_embeddings), author_ids

def compute_metrics(query_emb, target_emb, query_authors, target_authors, metric='cosine'):
    logging.info("Computing pairwise distances...")
    distances = pairwise_distances(query_emb.cpu(), target_emb.cpu(), metric=metric)
    ranks = np.zeros(len(query_authors), dtype=np.float32)
    reciprocal_ranks = np.zeros(len(query_authors), dtype=np.float32)
    logging.info("Evaluating rankings...")
    target_authors = np.array(target_authors)
    for i in tqdm(range(len(query_authors)), desc="Evaluating"):
        sorted_indices = np.argsort(distances[i])
        sorted_targets = target_authors[sorted_indices]
        match_indices = np.where(sorted_targets == query_authors[i])[0]
        ranks[i] = match_indices[0] if len(match_indices) else len(sorted_targets)
        reciprocal_ranks[i] = 1.0 / (ranks[i] + 1)

    return_dict = {
        "MRR": np.mean(reciprocal_ranks),
        "R@8": np.mean(ranks <= 8),
        "R@16": np.mean(ranks <= 16),
        "R@32": np.mean(ranks <= 32),
        "R@64": np.mean(ranks <= 64),
    }
    logging.info(f"\nResults:\n" + "\n".join(f"{k}: {v*100:.2f}" for k, v in return_dict.items()))

    return return_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["reddit", "amazon", "pan"], required=True)
    parser.add_argument(
        "--model", 
        choices=[
            "hf",
            "modernbert-reddit-single", "modernbert-reddit-all",
            "modernbert-wegmann-single", "modernbert-wegmann-all",
            "roberta-reddit-single", "roberta-reddit-all",
        ], required=True
    )
    parser.add_argument("--batch_size", type=int, default=3072)
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    MODEL_ROOT = os.path.join(BASE_DIR, "models")
    ROBERTA_ROOT = os.path.join(BASE_DIR, "models-roberta")
    MODEL_TEMPLATE = ["answerdotai", "ModernBERT-base-loss-triplet-margin-0.5-evaluator-triplet", "seed-1404"]
    def build_model_path(subdir, topic):
        return os.path.join(MODEL_ROOT, subdir, f"topic-{topic}", *MODEL_TEMPLATE)

    model_dict = {
        "hf": "AnnaWegmann/Style-Embedding",

        "modernbert-reddit-single":  build_model_path("modernbert_reddit_single_sentence_transformer_cache/av-models", "rand"),
        "modernbert-reddit-all":     build_model_path("modernbert_reddit_all_sentence_transformer_cache/av-models-all", "rand"),

        "modernbert-wegmann-single": build_model_path("modernbert_wegmann_single_sentence_transformer_cache/av-models", "conv"),
        "modernbert-wegmann-all":    build_model_path("modernbert_wegmann_all_sentence_transformer_cache/av-models-all", "conv"),

        "roberta-reddit-single": build_model_path("roberta_reddit_single_sentence_transformer_cache/av-models", "rand"),
        "roberta-reddit-all":    build_model_path("roberta_reddit_all_sentence_transformer_cache/av-models-all", "rand"),
    }
    print(model_dict)

    if args.model not in model_dict:
        raise ValueError(f"Unknown model key: {args.model}")
    model_str = model_dict[args.model]

    if args.dataset == "reddit":
        query_file = "raw_all/test_queries.jsonl"
        target_file = "raw_all/test_targets.jsonl"
    elif args.dataset == "amazon":
        query_file = "raw_amazon/validation_queries.jsonl"
        target_file = "raw_amazon/validation_targets.jsonl"
    elif args.dataset == "pan":
        query_file = "pan_paragraph/queries_raw.jsonl"
        target_file = "pan_paragraph/targets_raw.jsonl"

    logging.info("Loading and parsing queries...")
    query_texts, query_authors = load_texts_and_authors(args.dataset, query_file)

    logging.info("Loading and parsing targets...")
    target_texts, target_authors = load_texts_and_authors(args.dataset, target_file)

    logging.info("Encoding queries...")
    query_prefix = os.path.join(os.path.dirname(query_file), "encoded_queries")
    query_embs_raw, query_authors = encode_multi_gpu(query_texts, query_authors, args.batch_size, prefix_path=query_prefix, model_str=model_str)
    query_embeddings, query_authors = group_and_average(query_embs_raw, query_authors)

    logging.info("Encoding targets...")
    target_prefix = os.path.join(os.path.dirname(target_file), "encoded_targets")
    target_embs_raw, target_authors = encode_multi_gpu(target_texts, target_authors, args.batch_size, prefix_path=target_prefix, model_str=model_str)
    target_embeddings, target_authors = group_and_average(target_embs_raw, target_authors)

    logging.info("Computing metrics...")
    return_dict = compute_metrics(query_embeddings, target_embeddings, query_authors, target_authors)

    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["dataset", "model", "MRR", "R@8", "R@16", "R@32", "R@64"])

    with open(csv_path, mode='a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataset, args.model] + [f"{return_dict[k]*100:.2f}" for k in ['MRR','R@8','R@16','R@32','R@64']])

    if "all" in args.model:
        latex_row = (
            f"\t& {args.model} & "
            f"\\cellcolor{{gray!20}}{return_dict['MRR'] * 100:.2f} & "
            f"\\cellcolor{{gray!20}}{return_dict['R@8'] * 100:.2f} & "
            f"\\cellcolor{{gray!20}}{return_dict['R@16'] * 100:.2f} & "
            f"\\cellcolor{{gray!20}}{return_dict['R@32'] * 100:.2f} & "
            f"\\cellcolor{{gray!20}}{return_dict['R@64'] * 100:.2f} \\\\"
        )
    else:
        latex_row = (
            f"\t& {args.model} & "
            f"{return_dict['MRR'] * 100:.2f} & "
            f"{return_dict['R@8'] * 100:.2f} & "
            f"{return_dict['R@16'] * 100:.2f} & "
            f"{return_dict['R@32'] * 100:.2f} & "
            f"{return_dict['R@64'] * 100:.2f} \\\\"
        )
    print(args.dataset)
    print(latex_row)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Error running evaluation: {e}")
