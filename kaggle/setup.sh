#!/bin/bash
# One-time setup on Kaggle: packages, open-instruct, training script, data.
# Usage (from a Kaggle notebook):  !bash dr-lora/kaggle/setup.sh
set -euo pipefail

WORK=${WORK:-/kaggle/working}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OI_COMMIT=d2cf8a90d          # open-instruct, 26 Mar 2026 (transformers 4.x)
N_SAMPLES=${N_SAMPLES:-2000}  # training examples for the test run

echo "== 1. packages"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then
    pip install -q -r "$REPO/kaggle/requirements.txt"
    pip uninstall -y -q kernels 2>/dev/null || true   # breaks with hub 0.36
fi
pip list 2>/dev/null | grep -iE "^(torch|transformers|peft|accelerate|bitsandbytes|datasets) " || true

echo "== 2. open-instruct at $OI_COMMIT"
if [ ! -d "$WORK/open-instruct" ]; then
    git clone -q https://github.com/allenai/open-instruct.git "$WORK/open-instruct"
fi
git -C "$WORK/open-instruct" checkout -q "$OI_COMMIT"

echo "== 3. DR-LoRA training script (4-bit version) into open-instruct"
cp "$REPO/finetune/finetune.py" "$WORK/open-instruct/open_instruct/finetune_colm.py"
PYTHONPATH="$WORK/open-instruct" python -c "import open_instruct.finetune_colm; print('import OK')"

echo "== 4. data: MetaMathQA (GSM8K part), first $N_SAMPLES examples"
DATA="$WORK/data/metamathqa_gsm8k_${N_SAMPLES}.jsonl"
if [ ! -f "$DATA" ]; then
    python "$REPO/data/convert_metamathqa_to_tulu.py" --gsm8k_only \
        --max_samples "$N_SAMPLES" --output_path "$DATA"
fi
wc -l "$DATA"
echo "setup OK"
