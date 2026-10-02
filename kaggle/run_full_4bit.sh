#!/bin/bash
# Full-length DR-LoRA run on 4-bit OLMoE-1B-7B, both Kaggle T4s (~9 h).
# Paper schedule (3,750 steps, growth every 200, warm-up 3%, lr 2e-5,
# rank 8 -> 16, max 32) but batch 8 (4 per GPU) instead of 48:
# 30,000 examples (one pass), not 180,000. Saves a checkpoint every
# 250 steps; running this script again resumes from the latest one.
# Stops cleanly after TIME_LIMIT (default 11 h, Kaggle allows 12 h). To finish,
# start a new version with the old output added as input and set RESUME_FROM.
# Usage:  !bash dr-lora/kaggle/run_full_4bit.sh
#         !RESUME_FROM=/kaggle/input/<notebook>/out_drlora_4bit_full bash dr-lora/kaggle/run_full_4bit.sh
#         !NGPU=1 bash dr-lora/kaggle/run_full_4bit.sh   (old 1-GPU run, 15,000 examples)
set -euo pipefail

WORK=${WORK:-/kaggle/working}
REPO=$(cd "$(dirname "$0")/.." && pwd)
export NGPU=${NGPU:-2}
export N_SAMPLES=${N_SAMPLES:-$((15000 * NGPU))}   # 3,750 steps x 4 per GPU x NGPU
export DATA=${DATA:-$WORK/data/metamathqa_gsm8k_${N_SAMPLES}.jsonl}

if [ ! -s "$DATA" ]; then
    python "$REPO/data/convert_metamathqa_to_tulu.py" --gsm8k_only \
        --max_samples "$N_SAMPLES" --output_path "$DATA"
fi
echo "$(wc -l < "$DATA") training examples in $DATA"

OUT=${OUT:-$WORK/out_drlora_4bit_full}
# Part 2: copy the last finished checkpoint from the previous version's output
if [ -n "${RESUME_FROM:-}" ] && ! ls -d "$OUT"/step_* >/dev/null 2>&1; then
    # newest finished checkpoint (has a COMPLETED file)
    LAST=$(ls -d "$RESUME_FROM"/step_*/COMPLETED | sed 's|/COMPLETED$||' \
           | awk -F'step_' '{print $NF, $0}' | sort -n | tail -1 | cut -d' ' -f2-)
    echo "copying $LAST to $OUT"
    mkdir -p "$OUT"
    cp -r "$LAST" "$OUT/"
    [ -f "$RESUME_FROM/../train_full.log" ] && cp "$RESUME_FROM/../train_full.log" "$WORK/train_full.log"
fi

STEPS=${STEPS:-3750} GROW_EVERY=${GROW_EVERY:-200} WARMUP=${WARMUP:-0.03} \
LOG_EVERY=${LOG_EVERY:-10} CKPT_EVERY=${CKPT_EVERY:-250} RESUME=1 \
TIME_LIMIT=${TIME_LIMIT:-11h} OUT=$OUT LOG=${LOG:-$WORK/train_full.log} \
    bash "$REPO/kaggle/run_test_4bit.sh"
