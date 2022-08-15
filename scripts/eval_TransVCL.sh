#!/usr/bin/env bash
DATASET=VCSL
MODEL=model

# Before eval_TransVCL, please first execute test_TransVCL.sh and obtain pred_file.

python evaluation.py \
       --anno-file data/${DATASET}/label_file.json \
       --test-file data/${DATASET}/pair_file.csv \
       --pred-file results/${MODEL}/${DATASET}/result.json