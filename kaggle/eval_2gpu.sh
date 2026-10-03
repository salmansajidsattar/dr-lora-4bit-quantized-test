#!/bin/bash
# GSM8K on 2 GPUs: each GPU scores half of the 1,319 questions at the same
# time, then merge_eval.py combines the two halves into one score.
#
# Usage (Kaggle, GPU T4 x2):
#   !bash repo/kaggle/eval_2gpu.sh <run_dir>
#   !bash repo/kaggle/eval_2gpu.sh <run_dir> --no_adapter    # 4-bit base
# Progress: !tail -n 3 /kaggle/working/eval_part*.log
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RUN_DIR=${1:?give the training output folder}
shift
WORK=${WORK:-/kaggle/working}
BATCH=${BATCH:-8}

pids=()
for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i python "$HERE/eval_gsm8k_4bit.py" \
        --run_dir "$RUN_DIR" --part $i --parts 2 --batch_size $BATCH "$@" \
        > "$WORK/eval_part$i.log" 2>&1 &
    pids+=($!)
    echo "GPU $i started (log: $WORK/eval_part$i.log)"
done

ok=1
for i in 0 1; do
    wait ${pids[$i]} || { ok=0; echo "GPU $i FAILED, last lines:"; tail -n 20 "$WORK/eval_part$i.log"; }
done
[ $ok = 1 ] || exit 1

for i in 0 1; do tail -n 6 "$WORK/eval_part$i.log"; done
name=drlora_4bit
for a in "$@"; do [ "$a" = "--no_adapter" ] && name=base_4bit; done
python "$HERE/merge_eval.py" "$WORK/eval_gsm8k/${name}_part0of2.json" \
    "$WORK/eval_gsm8k/${name}_part1of2.json"
