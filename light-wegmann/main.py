"""
Default epochs is 20, but has early convergence logic - and it stops by 9 epochs in our trainings.
"""

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
from collections import defaultdict
from itertools import combinations
from pprint import pprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

print(torch.__version__)
print(torch.version.cuda)
torch.set_float32_matmul_precision('high')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
sys.path.append(os.path.join('', 'src'))
sys.path.append(os.path.join('', 'src/utils'))

from src.trainer_final_layer import SentenceBertFineTunerStandard
from src.trainer_all_layer import SentenceBertFineTunerAllLayers
from utils.convokit_generator import ANCHOR_COL, U1_COL, U2_COL, SAME_AUTHOR_AU1_COL
from utils.global_const import (
    SUBREDDIT_U2_COL, SUBREDDIT_U1_COL, SUBREDDIT_A_COL, 
    CONVERSATION_U2_COL, CONVERSATION_U1_COL, CONVERSATION_A_COL, 
    ID_U2_COL, ID_U1_COL, ID_A_COL, 
    AUTHOR_U2_COL, AUTHOR_U1_COL, AUTHOR_A_COL
)
from utils.training_const import TRIPLET_LOSS, TRIPLET_EVALUATOR


DEBUG = False
LUAR_REDDIT_BASE_PATH = "data/"
TRAIN_JSONL_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "data.jsonl")
VAL_QUERIES_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "validation_queries.jsonl")
VAL_TARGETS_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "validation_targets.jsonl")
TEST_QUERIES_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "test_queries.jsonl")
TEST_TARGETS_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "test_targets.jsonl")

LUAR_TRAIN_JSONL_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "luar_train.jsonl")
VAL_JSONL_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "luar_validation.jsonl")
TEST_JSONL_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "luar_test.jsonl")

# LOSS_FUNCTION = "contrastive"
LOSS_FUNCTION = TRIPLET_LOSS
# EVAL_TYPE="binary"
EVAL_TYPE=TRIPLET_EVALUATOR

MARGIN = 0.5  # Used for contrastive loss
# MAX_ENTRIES=200
#MAX_ENTRIES=12000
MAX_ENTRIES=30000
# MAX_ENTRIES=None
#MAX_PAIRS=500
MAX_PAIRS=None
MAX_PAIRS_PER_QUERY=1

# For Reddit Data (generated from LUAR's data via --generate_data)
LUAR_TRAIN_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "variable-random-luar_train-1.tsv")
LUAR_DEV_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "variable-random-luar_validation-1.tsv")
LUAR_TEST_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "variable-random-luar_test-1.tsv")

# For Wegmann data (**cleaned** inline newlines from original *-conversation.tsv from https://github.com/nlpsoc/Style-Embeddings/tree/master/Data/train_data)
WEGMANN_TRAIN_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "wegmann-train-variable-conversation-1.tsv")
WEGMANN_DEV_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "wegmann-dev-variable-conversation-1.tsv")
WEGMANN_TEST_TSV_PATH = os.path.join(LUAR_REDDIT_BASE_PATH, "wegmann-test-variable-conversation-1.tsv")


def clear_cuda():
    # Delete all variables that may be occupying GPU memory
    for obj in list(globals().keys()):
        if isinstance(globals()[obj], torch.Tensor):
            del globals()[obj]
    
    # Run garbage collection
    gc.collect()
    
    # Empty PyTorch CUDA cache
    torch.cuda.empty_cache()
    
    print("CUDA memory cleared!")


def stream_jsonl_reservoir_sample(file_path, max_entries=48000):
    """Reservoir sampling from a huge JSONL file, preserving DEBUG behavior and print structure."""
    if max_entries is None:
        return stream_jsonl(file_path)
    print(f"Sampling {max_entries} entries from {file_path} using reservoir sampling...")
    reservoir = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            data = json.loads(line)

            if len(reservoir) < max_entries:
                reservoir.append(data)
            else:
                j = random.randint(0, i)
                if j < max_entries:
                    reservoir[j] = data

            # Optional DEBUG printing — mimic original `stream_jsonl`
            if DEBUG and (len(reservoir) <= max_entries):
                print(f"Entry {i+1} Keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, str):
                        clipped_value = value[:200]
                    elif isinstance(value, list):
                        clipped_value = f"[List with {len(value)} elements] Example: {value[:2]}"
                    else:
                        clipped_value = value
                    print(f" * {key}: {clipped_value}")
                print("=" * 100)

    print(f"Completed sampling {len(reservoir)} entries.")
    return reservoir


def stream_jsonl(file_path, max_entries=None):
    """ Stream JSONL file and print structure with clipped values for debugging. """
    print(f"Streaming {file_path} (Clipped Output)...")
    entries = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f)):
            if max_entries is not None and i >= max_entries:
                break
            data = json.loads(line)

            if DEBUG:
                # Print keys and clipped values
                print(f"Entry {i+1} Keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, str):
                        clipped_value = value[:200]  # Clip long text fields
                    elif isinstance(value, list):
                        clipped_value = f"[List with {len(value)} elements] Example: {value[:2]}"  # Show first 2 elements
                    else:
                        clipped_value = value  # Print as is
                    print(f" * {key}: {clipped_value}")
                print("=" * 100)  
            
            entries.append(data)
    
    return entries

def generate_train_pairs(train_entries, max_pairs=None):
    if max_pairs is None:
        max_pairs = 2  # Default per-author limit

    author_sym_map = defaultdict(list)  # {author_id: [(text, action_type)]}
    action_author_map = defaultdict(list)  # {action_type: [(author_id, text)]}

    # Step 1: Organize Data
    print("# Step 1: Organize Data")
    for entry in tqdm(train_entries):
        syms = entry["syms"]
        action_types = entry["action_type"]
        author_id = entry["author_id"]

        if len(syms) != len(action_types):
            raise ValueError("Mismatched syms and action_type lengths within an entry")

        for i in range(len(syms)):
            author_sym_map[author_id].append((syms[i], action_types[i]))
            action_author_map[action_types[i]].append((author_id, syms[i]))

    contrastive_pairs = []

    # Step 2: Generate Pairs Per Author (Limited by max_pairs)
    print("Step 2: Generate Pairs Per Author (Limited by max_pairs)")
    for author, sym_list in tqdm(author_sym_map.items()):
        triplets_generated = 0  # Track per-author pairs

        # **Find (same author, different action) pairs**
        for (text1, action1), (text2, action2) in combinations(sym_list, 2):
            if action1 != action2:
                same_author_pair = (f"{text1} ", f" {text2} ")

                # **Find (different author, same action as text1) pairs**
                for other_author, other_text in action_author_map[action1]:  # Match action1
                    if other_author != author:
                        contrastive_pairs.append((same_author_pair[0], same_author_pair[1], 1, f"{other_text} "))
                        triplets_generated += 1
                        
                        if triplets_generated >= max_pairs:
                            break  # Stop after max_pairs per author
                
                if triplets_generated >= max_pairs:
                    break  # Stop after max_pairs per author

    random.shuffle(contrastive_pairs)

    print(f"Generated {len(contrastive_pairs)} training contrastive triplets.")
    #print(contrastive_pairs[:5])  # Print first 5 pairs for validation
    return contrastive_pairs

def extract_query_target_pairs(queries, targets, max_pairs_per_query=None):
    if max_pairs_per_query is None:
        max_pairs_per_query=MAX_PAIRS_PER_QUERY

    """
    Generate triplets: ( author1-text1, author1-text2, otherAuthor-text3, 1 )
    
    Constraints:
      - text1, text2: same author, different action_type
      - text3: different author, same action_type as text2
      - up to max_pairs_per_query triplets per query
      - avoid reusing the same text2 for a query
    """

    # 1) Build a lookup such that for each author and each action_type,
    #    we have a list of possible texts (syms).
    target_dict = defaultdict(lambda: defaultdict(list))
    print("for t in tqdm(targets):")
    for t in tqdm(targets):
        # Expect: t has keys "author_id", "syms", "action_type"
        # length(t["syms"]) == length(t["action_type"])
        if ("author_id" in t and 
            "syms" in t and 
            "action_type" in t and 
            len(t["syms"]) == len(t["action_type"])):
            
            author = t["author_id"]
            for sym, act in zip(t["syms"], t["action_type"]):
                target_dict[author][act].append(sym)

    contrastive_pairs = []

    # 2) Iterate over each query
    print("2) Iterate over each query")
    for query in tqdm(queries):
        if ("author_id" not in query or
            "syms" not in query or
            "action_type" not in query):
            continue  # Skip invalid query items

        author_id = query["author_id"]
        query_syms = query["syms"]
        query_actions = query["action_type"]

        # Sanity check
        if len(query_syms) != len(query_actions):
            continue

        triplets_generated = 0
        used_text2 = set()

        # 3) Generate (text1, text2) pairs from the same query
        for i in range(len(query_syms)):
            for j in range(i + 1, len(query_syms)):
                if triplets_generated >= max_pairs_per_query:
                    break

                text1, act1 = query_syms[i], query_actions[i]
                text2, act2 = query_syms[j], query_actions[j]

                # Must have different action types
                if act1 == act2:
                    continue
                
                # Avoid re-using text2 for this query
                if text2 in used_text2:
                    continue
                used_text2.add(text2)

                # 4) Find text3 from a different author, but with the same action_type as text2
                for other_author, author_dict in target_dict.items():
                    if other_author == author_id:
                        continue  # must be a different author

                    # Must match act2
                    if act2 in author_dict:
                        for txt3 in author_dict[act2]:
                            contrastive_pairs.append((
                                f"{text1} ",
                                f"{text2} ",
                                1,
                                f"{txt3} "
                            ))
                            triplets_generated += 1

                            if triplets_generated >= max_pairs_per_query:
                                break

                    if triplets_generated >= max_pairs_per_query:
                        break

            if triplets_generated >= max_pairs_per_query:
                break

    # Shuffle the overall list to avoid ordering bias
    random.shuffle(contrastive_pairs)
    print(f"Generated {len(contrastive_pairs)} triplets in total.")

    return contrastive_pairs

def save_pairs_to_jsonl(pairs, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for text1, text2, text3, label in pairs:
            json.dump({"text1": text1, "text2": text2, "label": label, "text3": text3}, f)
            f.write("\n")  # Ensure each entry is on a new line
    #print(f"Saved {len(pairs)} pairs to {file_path}")

def save_to_tsv(pairs, output_tsv_path):
    def substitute_string(phrase):
        return phrase.replace('"', '\\"').replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")

    """ Saves contrastive pairs to a TSV file with proper formatting. """
    # print("\n\n")
    # pprint(pairs)
    df = pd.DataFrame(pairs, columns=["Anchor (A)", "Utterance 1 (U1)", "Same Author Label", "Utterance 2 (U2)"])
    #df["Utterance 2 (U2)"] = df["Utterance 1 (U1)"]  # Duplicate U1 for U2

    # Ensure newlines are escaped properly
    df["Anchor (A)"] = df["Anchor (A)"].apply(lambda x: substitute_string(x))
    df["Utterance 1 (U1)"] = df["Utterance 1 (U1)"].apply(lambda x: substitute_string(x))
    df["Utterance 2 (U2)"] = df["Utterance 2 (U2)"].apply(lambda x: substitute_string(x))

    # Save as TSV (with quoting=3 for safety)
    df.to_csv(output_tsv_path, sep="\t", index=False, escapechar="\\", quoting=3)
    #print(f"Saved {len(df)} samples to {output_tsv_path}")

def load_test(task_filename):
    try:
        return pd.read_csv(task_filename, sep='\t',
                            dtype={ANCHOR_COL: str, U1_COL: str, U2_COL: str,
                                    ID_U1_COL: str, ID_U2_COL: str, ID_A_COL: str,
                                    AUTHOR_A_COL: str, AUTHOR_U2_COL: str, AUTHOR_U1_COL: str,
                                    CONVERSATION_U1_COL: str, CONVERSATION_U2_COL: str, CONVERSATION_A_COL: str,
                                    SUBREDDIT_A_COL: str, SUBREDDIT_U1_COL: str, SUBREDDIT_U2_COL: str,
                                    SAME_AUTHOR_AU1_COL: int})
    except Exception as e:
        print(f"[Fallback] Standard TSV loading failed: {e}")
        df = pd.read_csv(
            task_filename,
            sep='\t',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            engine='python'
        )

        def clean(s):
            if isinstance(s, str):
                return (
                    s.replace("\\n", " ")
                     .replace("\\t", " ")
                     .replace("\\r", " ")
                     .replace('\\"', '"')
                     .strip()
                )
            return s

        for col in ["Anchor (A)", "Utterance 1 (U1)", "Utterance 2 (U2)"]:
            if col in df.columns:
                df[col] = df[col].apply(clean)

        return df

def create_train():
    print("Process training data from `data.jsonl`...")
    train_entries = stream_jsonl_reservoir_sample(TRAIN_JSONL_PATH, max_entries=MAX_ENTRIES)
    train_pairs = generate_train_pairs(train_entries, max_pairs=MAX_PAIRS)
    save_pairs_to_jsonl(train_pairs, LUAR_TRAIN_JSONL_PATH)
    print(f"Training pairs saved to {LUAR_TRAIN_JSONL_PATH}")
    save_to_tsv(train_pairs, TRAIN_TSV_PATH)
    print(f"Training TSV saved to {TRAIN_TSV_PATH}")

def create_val():
    # val_queries = stream_jsonl(VAL_QUERIES_PATH, max_entries=None)
    val_queries = stream_jsonl(VAL_QUERIES_PATH, max_entries=MAX_ENTRIES)
    # val_queries = stream_jsonl_reservoir_sample(VAL_QUERIES_PATH, max_entries=MAX_ENTRIES)
    print("VAL_QUERIES_PATH", VAL_QUERIES_PATH, len(val_queries))
    # val_targets = stream_jsonl(VAL_TARGETS_PATH, max_entries=None)
    val_targets = stream_jsonl(VAL_TARGETS_PATH, max_entries=MAX_ENTRIES)
    # val_targets = stream_jsonl_reservoir_sample(VAL_TARGETS_PATH, max_entries=MAX_ENTRIES)
    print("VAL_TARGETS_PATH", VAL_TARGETS_PATH, len(val_targets))
    val_pairs = extract_query_target_pairs(val_queries, val_targets)
    print(f"\nGenerated {len(val_pairs)} validation pairs.")
    save_pairs_to_jsonl(val_pairs, VAL_JSONL_PATH)
    save_to_tsv(val_pairs, DEV_TSV_PATH)
    print(f"Validation TSV saved to {DEV_TSV_PATH}")

def create_test():
    # test_queries = stream_jsonl(TEST_QUERIES_PATH, max_entries=None)
    test_queries = stream_jsonl(TEST_QUERIES_PATH, max_entries=MAX_ENTRIES)
    # test_queries = stream_jsonl_reservoir_sample(TEST_QUERIES_PATH, max_entries=MAX_ENTRIES)
    print("TEST_QUERIES_PATH", TEST_QUERIES_PATH, len(test_queries))
    # test_targets = stream_jsonl(TEST_TARGETS_PATH, max_entries=None)
    test_targets = stream_jsonl(TEST_TARGETS_PATH, max_entries=MAX_ENTRIES)
    # test_targets = stream_jsonl_reservoir_sample(TEST_TARGETS_PATH, max_entries=MAX_ENTRIES)
    print("TEST_TARGETS_PATH", TEST_TARGETS_PATH, len(test_targets))
    test_pairs = extract_query_target_pairs(test_queries, test_targets)
    print(f"Generated {len(test_pairs)} test pairs.")
    save_pairs_to_jsonl(test_pairs, TEST_JSONL_PATH)
    save_to_tsv(test_pairs, TEST_TSV_PATH)
    print(f"Test TSV saved to {TEST_TSV_PATH}")

def test_datasets():
    task_filename = "data/variable-random-luar_train.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

    task_filename = "data/variable-random-luar_validation.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

    task_filename = "data/variable-random-luar_test.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

def test_golden_datasets():
    task_filename = "data/wegmann-train-variable-conversation.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

    task_filename = "data/wegmann-dev-variable-conversation.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

    task_filename = "data/wegmann-test-variable-conversation.tsv"
    print("task_filename", task_filename)
    load_test(task_filename)

def create_trainer(mode, cache_folder, train_tsv, dev_tsv, backbone):
    if mode == 'single':
        TrainerClass = SentenceBertFineTunerStandard
        print("Initializing SentenceBertFineTunerStandard")
    elif mode == 'all':
        TrainerClass = SentenceBertFineTunerAllLayers
        print("Initializing SentenceBertFineTunerAllLayers")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    tuner = TrainerClass(
        model_path=backbone,
        train_filename=train_tsv,
        dev_filename=dev_tsv,
        loss=LOSS_FUNCTION,
        margin=MARGIN,
        evaluation_type=EVAL_TYPE,
        cache_folder=cache_folder,
        seed=1404
    )
    print("Fine-Tuner Initialized Successfully")

    return tuner

def training(tuner, epochs, batch_size):
    clear_cuda()
    print("Starting fine-tuning")
    save_dir = tuner.train(
        epochs=epochs,
        batch_size=batch_size
    )
    print(f"Training completed! Model saved at: {save_dir}")

    return save_dir

def patch_modules_json(save_dir):
    modules_path = os.path.join(save_dir, "modules.json")

    if not os.path.exists(modules_path):
        print(f"[patch_modules_json] modules.json not found at {modules_path}")
        return

    with open(modules_path, "r") as f:
        modules = json.load(f)

    patched = False
    for module in modules:
        if module["type"] == "src.trainer_all_layer.TransformerAllLayers":
            print("[patch_modules_json] Rewriting module type to sentence_transformers.models.Transformer")
            module["type"] = "sentence_transformers.models.Transformer"
            patched = True

    if patched:
        with open(modules_path, "w") as f:
            json.dump(modules, f, indent=2)
        print(f"[patch_modules_json] Patched modules.json saved to {modules_path}")
    else:
        print("[patch_modules_json] No changes needed.")


def inference(tuner, test_tsv):
    print(f"Loading test dataset from: {test_tsv}...")
    test_df = pd.read_csv(test_tsv, sep="\t")

    print("Running inference on test dataset...")
    test_pairs = list(zip(test_df["Anchor (A)"], test_df["Utterance 1 (U1)"]))

    results = []
    for text1, text2 in test_pairs[:10]:  # Limit to 10 samples for quick verification
        with torch.no_grad():
            score = tuner.similarity(text1, text2)  # Using the fine-tuned model
        results.append((text1, text2, score.item()))

    print("\nInference Completed! Sample Results:")
    for text1, text2, score in results:
        print(f"Text 1: {text1[:50]}...")
        print(f"Text 2: {text2[:50]}...")
        print(f"Similarity Score: {score:.4f}")
        print("=" * 60)

    print("\nInference Completed on Test Set!")

    print("[Results] Computing similarity scores for the full test set...")

    # Prepare text pairs and ground-truth labels
    test_pairs = list(zip(test_df["Anchor (A)"], test_df["Utterance 1 (U1)"]))
    ground_truth_labels = test_df["Same Author Label"].tolist()

    # Compute similarity scores
    predicted_scores = []

    for text1, text2 in tqdm(test_pairs, desc="Evaluating", unit="pair"):
        with torch.no_grad():
            score = tuner.similarity(text1, text2)
        predicted_scores.append(score.item())

    # Convert scores to binary labels (Threshold = 0.5 for same/different author classification)
    predicted_labels = [1 if score >= 0.5 else 0 for score in predicted_scores]

    print("[Results] Similarity scores computed!")

    # Compute evaluation metrics
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth_labels, predicted_labels, average="binary")

    # Display results
    print("\nEvaluation Metrics on Test Set")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nEvaluation Completed!")


def main(args):
    if args.generate_data is True:
        print("Generating data")
        create_train()
        create_val()
        create_test()
        test_datasets()
        exit(0)

    # test_golden_datasets()
    # exit(0)

    location_prefix = ""

    if args.backbone == "modernbert":
        MODEL_NAME = "answerdotai/ModernBERT-base"
        location_prefix = location_prefix + "modernbert_"
    else:
        MODEL_NAME = "roberta-base"
        location_prefix = location_prefix + "roberta_"

    if args.data == "wegmann":
        location_prefix = location_prefix + "wegmann_"
        TRAIN_TSV_PATH = WEGMANN_TRAIN_TSV_PATH
        DEV_TSV_PATH = WEGMANN_DEV_TSV_PATH
        TEST_TSV_PATH = WEGMANN_TEST_TSV_PATH
    else:
        location_prefix = location_prefix + "reddit_"
        TRAIN_TSV_PATH = LUAR_TRAIN_TSV_PATH
        DEV_TSV_PATH = LUAR_DEV_TSV_PATH
        TEST_TSV_PATH = LUAR_TEST_TSV_PATH

    tuner = create_trainer(
                mode=args.mode,
                cache_folder=f"models/{location_prefix}{args.mode}_sentence_transformer_cache/",
                train_tsv=TRAIN_TSV_PATH,
                dev_tsv=DEV_TSV_PATH,
                backbone=MODEL_NAME,
            )
    save_dir = training(tuner, args.epochs, args.batch_size)

    if args.mode == "all":
        """
        Replace FROM  "type": "src.trainer_all_layer.TransformerAllLayers"
        TO "type": "sentence_transformers.models.Transformer"
        in modules.json of the saved model when using our custom class
        to allow loading and eval using default SBERT class
        """
        patch_modules_json(save_dir)

    print(f"save_dir: {save_dir}")
    # inference(tuner, test_tsv=TEST_TSV_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["modernbert", "roberta"], required=True,
                        help="Pretrained model base to be used")
    parser.add_argument("--data", type=str, choices=["reddit", "wegmann"], required=True,
                        help="Data used to train: luar's reddit or wegmann's conv. IMP: For luar's reddit, first please use --generate-data then start using")
    parser.add_argument("--mode", type=str, choices=["single", "all"], required=True,
                        help="Training mode to use: single or all")
    parser.add_argument("--generate_data", type=bool, default=False, help="If set, generates TSV/jsonl datasets")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=24)
    args = parser.parse_args()

    main(args)
