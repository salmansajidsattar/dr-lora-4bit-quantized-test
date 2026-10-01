#!/bin/bash
# ================= Configuration =================

# ===== Model Configuration =====
# Format: one model per line, fields separated by |
#   MODEL_PATH | BASE_OUTPUT_DIR
#
# Comment/uncomment lines to enable/disable models.
MODEL_CONFIGS=(
    "/path/to/pretrained_models/Qwen1.5-MoE-A2.7B | /path/to/results/qwen1.5-moe-a2.7b/base/medmcqa"
    "/path/to/output/qwen_medmcqa_drlora_r8_target16/step_3750/merged_model | /path/to/results/qwen1.5-moe-a2.7b/qwen_medmcqa_drlora_r8_target16/step_3750"
)

DATA_PATH="/path/to/data/MedMCQA/test.json"

CUDA_DEVICE=0

# ===== Multi-run Configuration =====
NUM_RUNS=3
RUN_SEEDS=(42 123 456)    # Must match NUM_RUNS in length

# ===== Quick Test Configuration =====
QUICK_TEST=false
TEST_LIMIT=50     # Number of samples when QUICK_TEST=true

# ==========================================

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

EVAL_SCRIPT="$(dirname "$0")/eval_medmcqa.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[Error] Eval script not found: $EVAL_SCRIPT"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "[Error] Data file not found: $DATA_PATH"
    exit 1
fi

if [ "$QUICK_TEST" = true ]; then
    LIMIT_ARG="--max_samples $TEST_LIMIT"
    echo "QUICK TEST MODE: evaluating first $TEST_LIMIT samples only"
else
    LIMIT_ARG=""
    echo "FULL TEST MODE: evaluating all samples"
fi

# ---------------------------------------------------------------------------
# compute_stats — compute mean and std dev across runs for a single model
# ---------------------------------------------------------------------------
compute_stats() {
    local MODEL_PATH=$1
    local BASE_OUTPUT_DIR=$2

    local RESULT_FILES=()
    for RUN_IDX in $(seq 1 $NUM_RUNS); do
        local F="$BASE_OUTPUT_DIR/run_${RUN_IDX}/results_summary.json"
        if [ -f "$F" ]; then
            RESULT_FILES+=("$F")
        else
            echo "  Run $RUN_IDX result file not found: $F"
        fi
    done

    if [ ${#RESULT_FILES[@]} -eq 0 ]; then
        echo "  No result files found, skipping stats."
        return
    fi

    local SUMMARY_DIR="$BASE_OUTPUT_DIR/summary"
    mkdir -p "$SUMMARY_DIR"

    python3 - "${RESULT_FILES[@]}" "$SUMMARY_DIR" <<'PYEOF'
import sys, json, math, os

result_files = sys.argv[1:-1]
summary_dir  = sys.argv[-1]

all_runs = []
for fpath in result_files:
    with open(fpath) as f:
        all_runs.append(json.load(f))

if not all_runs:
    print("  No valid result files found.")
    sys.exit(1)

model_path = all_runs[0].get("model", "unknown")

metrics_to_track = ["overall", "single", "multi"]
collected = {m: [] for m in metrics_to_track}

for run_data in all_runs:
    acc = run_data.get("accuracy", {})
    for m in metrics_to_track:
        if m in acc:
            collected[m].append(float(acc[m]))

def calc_stats(vals):
    n = len(vals)
    if n == 0:
        return {"values": [], "mean": None, "std": None, "n_runs": 0}
    mu = sum(vals) / n
    std = math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"values": [round(v, 6) for v in vals], "mean": round(mu, 6), "std": round(std, 6), "n_runs": n}

stats = {m: calc_stats(collected[m]) for m in metrics_to_track}

output = {
    "model":   model_path,
    "n_runs":  len(result_files),
    "results": stats,
}

out_path = os.path.join(summary_dir, "medmcqa_stats.json")
with open(out_path, "w") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"  Stats saved to: {out_path}")
print(f"\n  {'Metric':<12} {'Mean':>10}  {'Std':>10}  {'Values'}")
print(f"  {'-'*12} {'-'*10}  {'-'*10}  {'-'*30}")
for m, s in stats.items():
    if s["mean"] is None:
        continue
    vals_str = "  ".join(f"{v:.4f}" for v in s["values"])
    print(f"  {m:<12} {s['mean']*100:>9.2f}%  {s['std']*100:>9.2f}%  [{vals_str}]")
PYEOF
}

# ---------------------------------------------------------------------------
# run_model — run all runs for a single model and compute stats
# ---------------------------------------------------------------------------
run_model() {
    local MODEL_PATH=$1
    local BASE_OUTPUT_DIR=$2

    echo ""
    echo "###################################################################"
    echo "###  MODEL: $(basename $MODEL_PATH)"
    echo "###  OUTPUT: $BASE_OUTPUT_DIR"
    echo "###################################################################"
    echo ""

    mkdir -p "$BASE_OUTPUT_DIR"

    for RUN_IDX in $(seq 1 $NUM_RUNS); do
        local SEED=${RUN_SEEDS[$((RUN_IDX - 1))]}
        local RUN_OUTPUT_DIR="$BASE_OUTPUT_DIR/run_${RUN_IDX}"
        mkdir -p "$RUN_OUTPUT_DIR"

        echo ""
        echo "--- RUN $RUN_IDX / $NUM_RUNS  (seed=$SEED) ---"
        echo ""

        local RUN_START=$(date +%s)

        python "$EVAL_SCRIPT" \
            --model_path "$MODEL_PATH" \
            --data_path  "$DATA_PATH" \
            --output_path "$RUN_OUTPUT_DIR" \
            --gpu "$CUDA_DEVICE" \
            --seed "$SEED" \
            $LIMIT_ARG

        local RUN_END=$(date +%s)
        local RUN_ELAPSED=$((RUN_END - RUN_START))
        echo "  Run $RUN_IDX complete, elapsed $((RUN_ELAPSED/60))m $((RUN_ELAPSED%60))s"
    done

    echo ""
    echo "--- Computing Mean and Std Dev for $(basename $MODEL_PATH) ---"
    echo ""
    compute_stats "$MODEL_PATH" "$BASE_OUTPUT_DIR"
}

# ---------------------------------------------------------------------------
# Main: iterate over all models
# ---------------------------------------------------------------------------
TOTAL_START=$(date +%s)

echo "========================================================================"
echo "MedMCQA Batch Evaluation — ${#MODEL_CONFIGS[@]} model(s) x ${NUM_RUNS} runs"
echo "Data   : $DATA_PATH"
echo "GPU    : $CUDA_DEVICE"
echo "Quick Test: $QUICK_TEST"
echo "========================================================================"

MODEL_IDX=0
for MODEL_CFG in "${MODEL_CONFIGS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))

    IFS='|' read -r MODEL_PATH BASE_OUTPUT_DIR <<< "$MODEL_CFG"
    MODEL_PATH=$(echo "$MODEL_PATH"           | xargs)
    BASE_OUTPUT_DIR=$(echo "$BASE_OUTPUT_DIR" | xargs)

    echo ""
    echo "###################################################################"
    echo "###  MODEL $MODEL_IDX / ${#MODEL_CONFIGS[@]}"
    echo "###################################################################"

    MODEL_START=$(date +%s)
    run_model "$MODEL_PATH" "$BASE_OUTPUT_DIR"
    MODEL_END=$(date +%s)

    MODEL_ELAPSED=$((MODEL_END - MODEL_START))
    echo ""
    echo "Model $(basename $MODEL_PATH) finished in $((MODEL_ELAPSED/60))m $((MODEL_ELAPSED%60))s"

    STATS_FILE="$BASE_OUTPUT_DIR/summary/medmcqa_stats.json"
    if [ -f "$STATS_FILE" ]; then
        echo ""
        echo "Quick Summary — $(basename $MODEL_PATH) (Mean% +/- Std%):"
        python3 -c "
import json
with open('$STATS_FILE') as f:
    d = json.load(f)
for metric, s in d['results'].items():
    if s['mean'] is not None:
        print(f\"  {metric:<12} {s['mean']*100:.2f}% +/- {s['std']*100:.2f}%\")
"
    fi
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "========================================================================"
echo "All Models Completed in $((TOTAL_ELAPSED/60))m $((TOTAL_ELAPSED%60))s"
echo "========================================================================"