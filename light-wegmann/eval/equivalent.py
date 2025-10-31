import argparse
from evaluate import load_texts_and_authors, group_and_average, load_jsonl
from collections import Counter
from tqdm import tqdm
import numpy as np
import os
import torch
import traceback

def check_dataset_consistency(name, query_path, target_path):
    print(f"\n=== CHECKING: {name.upper()} ===")

    # Load raw
    query_texts, query_authors = load_texts_and_authors(name, query_path)
    target_texts, target_authors = load_texts_and_authors(name, target_path)

    # Count raw appearances
    qc = Counter(query_authors)
    tc = Counter(target_authors)

    # Check 1: At least one duplicate author
    mult_q = [a for a, n in qc.items() if n > 1]
    mult_t = [a for a, n in tc.items() if n > 1]
    print(f"Query authors with multiple texts: {len(mult_q)} / {len(qc)}")
    print(f"Target authors with multiple texts: {len(mult_t)} / {len(tc)}")

    # Check 2: Unique author counts after grouping
    # q_embs = [np.random.randn(768)] * len(query_authors)
    # t_embs = [np.random.randn(768)] * len(target_authors)
    q_embs = [torch.randn(768) for _ in query_authors]
    t_embs = [torch.randn(768) for _ in target_authors]
    q_avg, q_auth = group_and_average(q_embs, query_authors)
    t_avg, t_auth = group_and_average(t_embs, target_authors)
    print(f"Unique query authors after averaging: {len(set(q_auth))}")
    print(f"Unique target authors after averaging: {len(set(t_auth))}")

    # Check 3: All query authors exist in targets
    q_set, t_set = set(q_auth), set(t_auth)
    missing = q_set - t_set
    if missing:
        print(f"{len(missing)} query authors missing in targets! Example: {list(missing)[:3]}")
    else:
        print("All query authors are present in targets")

    # Check 4: Author ID consistency
    q_types = {type(a) for a in q_auth}
    t_types = {type(a) for a in t_auth}
    if q_types != t_types:
        print(f"Mismatched author ID types: queries={q_types}, targets={t_types}")
    else:
        print(f"Author ID types consistent: {list(q_types)[0]}")

    print("")

def inspect_syms_field(dataset_name, file_path, num_samples=5):
    print(f"\n[Inspecting `{dataset_name}` @ {file_path}]")
    data = load_jsonl(file_path)
    for i, entry in enumerate(data[:num_samples]):
        syms = entry.get("syms")
        print(f"Entry {i}:")
        print(f"  author_id: {entry.get('author_id')}")
        print(f"  type(syms): {type(syms)}")
        if isinstance(syms, list):
            print(f"  len(syms): {len(syms)}")
            print(f"  first item: {repr(syms[0])}")
            if isinstance(syms[0], dict):
                print(f"    first item keys: {list(syms[0].keys())}")
        else:
            print("  [WARNING] 'syms' is not a list!")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)

    args = parser.parse_args()
    root = args.root

    datasets = {
        "reddit": (
            os.path.join(root, "raw_all/test_queries.jsonl"),
            os.path.join(root, "raw_all/test_targets.jsonl"),
        ),
        "amazon": (
            os.path.join(root, "raw_amazon/validation_queries.jsonl"),
            os.path.join(root, "raw_amazon/validation_targets.jsonl"),
        ),
        "pan": (
            os.path.join(root, "pan_paragraph/queries_raw.jsonl"),
            os.path.join(root, "pan_paragraph/targets_raw.jsonl"),
        ),
    }

    inspect_syms_field("reddit", "./raw_all/test_queries.jsonl")
    inspect_syms_field("amazon", "./raw_amazon/validation_queries.jsonl")
    inspect_syms_field("pan", "./pan_paragraph/queries_raw.jsonl")
    # exit(0)

    for name, (qf, tf) in datasets.items():
        try:
            check_dataset_consistency(name, qf, tf)
        except Exception as exc:

            print(traceback.format_exc())
            print(exc)
