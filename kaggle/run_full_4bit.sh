#!/bin/bash
# Full-length DR-LoRA run on 4-bit OLMoE-1B-7B, one Kaggle T4 (~9 h).
# Paper schedule (3,750 steps, growth every 200, warm-up 3%, lr 2e-5,
# rank 8 -> 16, max 32) but batch 4 instead of 48: 15,000 examples
# (one pass), not 180,000. Saves a checkpoint every 250 steps; running
# this script again resumes from the latest one.
# Usage:  !bash dr-lora/kaggle/run_full_4bit.sh
set -euo pipefail

WORK=${WORK:-/kaggle/working}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export N_SAMPLES=${N_SAMPLES:-15000}
export DATA=${DATA:-$WORK/data/metamathqa_gsm8k_${N_SAMPLES}.jsonl}

if [ ! -s "$DATA" ]; then
    python "$REPO/data/convert_metamathqa_to_tulu.py" --gsm8k_only \
        --max_samples "$N_SAMPLES" --output_path "$DATA"
fi
echo "$(wc -l < "$DATA") training examples in $DATA"

STEPS=${STEPS:-3750} GROW_EVERY=${GROW_EVERY:-200} WARMUP=${WARMUP:-0.03} \
LOG_EVERY=${LOG_EVERY:-10} CKPT_EVERY=${CKPT_EVERY:-250} RESUME=1 \
OUT=${OUT:-$WORK/out_drlora_4bit_full} LOG=${LOG:-$WORK/train_full.log} \
    bash "$REPO/kaggle/run_test_4bit.sh"
