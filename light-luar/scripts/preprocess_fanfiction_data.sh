# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash
# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers. 
#
# SPDX-License-Identifier: Apache-2.0

#! /usr/bin/env bash


if [ ! -d "/mnt/swordfish-pool2/nikhil/pan_paragraph" ]; then
    mkdir /mnt/swordfish-pool2/nikhil/pan_paragraph
fi

data_path="/mnt/swordfish-pool2/nikhil/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large.jsonl"
truth_path="/mnt/swordfish-pool2/nikhil/pan20-authorship-verification-training-large/pan20-authorship-verification-training-large-truth.jsonl"

python scripts/preprocess_pan_into_paragraphs.py ${data_path} ${truth_path}
