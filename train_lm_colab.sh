#!/bin/bash

lr=1e-3

run_name='1e-3_test'
out_dir="checkpoints/${run_name}"

uv run cs336_basics/train/train_llm.py \
    --lr "$lr" \
    --out_dir "$out_dir" \
    --train_bin "/content/llm_from_scratch/data" \
    --val_bin "/content/llm_from_scratch/data" \
    --max_steps 80000 \
    --eval_interval 200 \
    --warmup_steps 1000 \
    --log_interval 100 \
    --wandb \
    --wandb_project "cs336-training-tinystory" \
    --wandb_run_name "$run_name" \


echo "Finished run for LR: $lr"