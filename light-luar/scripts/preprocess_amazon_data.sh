# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash

if [ ! -d "/mnt/swordfish-pool2/nikhil/raw_amazon" ]; then
    mkdir /mnt/swordfish-pool2/nikhil/raw_amazon
fi

data_path="/mnt/swordfish-pool2/nikhil/All_Amazon_Review.json.gz"
stats_path="/mnt/swordfish-pool2/nikhil/stats.npy"

python scripts/process_amazon_rawtxt.py ${data_path} ${stats_path}

head -n 100000 /mnt/swordfish-pool2/nikhil/raw_amazon/out_raw_100.jsonl > /mnt/swordfish-pool2/nikhil/raw_amazon/train.jsonl
tail -n 35059 /mnt/swordfish-pool2/nikhil/raw_amazon/out_raw_100.jsonl > /mnt/swordfish-pool2/nikhil/raw_amazon/validation.jsonl

python scripts/split_amazon.py
