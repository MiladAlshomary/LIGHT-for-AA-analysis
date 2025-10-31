#!/bin/bash
set -x
set -e
mkdir -p logs

# ModernBERT backbone
# Reddit
time python main.py --backbone modernbert --data reddit --mode single 2>&1 | tee logs/modernbert_reddit_single_training.txt
time python main.py --backbone modernbert --data reddit --mode all 2>&1 | tee logs/modernbert_reddit_all_training.txt

# Wegmann
time python main.py --backbone modernbert --data wegmann --mode single --batch_size 16 2>&1 | tee -a logs/modernbert_wegmann_single_training.txt
time python main.py --backbone modernbert --data wegmann --mode all --batch_size 16 2>&1 | tee -a logs/modernbert_wegmann_all_training.txt

# RoBERTa backbone
# Reddit
time python main.py --backbone roberta --data reddit --mode single 2>&1 | tee logs/roberta_reddit_single_training.txt
time python main.py --backbone roberta --data reddit --mode all 2>&1 | tee logs/roberta_reddit_all_training.txt

# Evaluate
DATASETS=("pan" "reddit" "amazon")
MODELS=("hf" "modernbert-reddit-single" "modernbert-reddit-all" "modernbert-wegmann-single" "modernbert-wegmann-all" "roberta-reddit-single" "roberta-reddit-all")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    log_file="logs/eval_${dataset}_${model}.log"
    echo "Running: Dataset=$dataset, Model=$model"
    echo "Logging to: $log_file"
    echo "=============================="

    python eval/evaluate.py \
      --dataset "$dataset" \
      --model "$model" \
      --batch_size 128 \
      2>&1 | tee -a "$log_file"

    echo "------------------------------"
    echo ""
  done
done
set +x