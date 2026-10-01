#!/bin/bash
# ================= Configuration =================

# ===== Model Configuration =====
# Format: one model per line, fields separated by |
#   MODEL_PATH | BASE_OUTPUT_DIR | MODEL_TYPE
#
# MODEL_TYPE options:
#   base     -> pretrained base model; script adds --force-base-prompt automatically
#   instruct -> fine-tuned model; evalplus auto-detects chat_template
#
# Comment/uncomment lines to enable/disable models.
MODEL_CONFIGS=(
    "/path/to/pretrained_models/Qwen1.5-MoE-A2.7B | /path/to/results/qwen1.5-moe-a2.7b/base | base"
    "/path/to/output/qwen_humaneval_codealpaca_lora_baseline_r32/step_3750/merged_model | /path/to/results/qwen1.5-moe-a2.7b/qwen_humaneval_codealpaca_lora_baseline_r32/step_3750 | instruct"
)

CUDA_DEVICE=0

# ===== Multi-run Configuration =====
NUM_RUNS=3

# Random seed for each run (must match NUM_RUNS in length)
RUN_SEEDS=(42 123 456)

# ==========================================
# ===== Task Configuration =====
# Comment/uncomment lines to enable/disable tasks.
TASK_CONFIGS=(
    "humaneval"
#    "mbpp"
)

# ==========================================

export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# run_evalplus_with_seed.py should be placed in the same directory as this script
WRAPPER_SCRIPT="$(dirname "$0")/run_evalplus_with_seed.py"

# ==========================================
# run_evalplus DATASET RUN_IDX
#   DATASET : humaneval | mbpp
#   RUN_IDX : run index (1-based)
#   MODEL_PATH / BASE_OUTPUT_DIR / TYPE_ARGS are set by the outer model loop
# ==========================================
run_evalplus() {
    local DATASET=$1
    local RUN_IDX=$2
    local SEED=${RUN_SEEDS[$((RUN_IDX - 1))]}

    local RUN_DIR="$BASE_OUTPUT_DIR/run_${RUN_IDX}"
    mkdir -p "$RUN_DIR"

    echo "========================================================================"
    echo "Running EvalPlus"
    echo "  Dataset    : ${DATASET}+"
    echo "  Run        : $RUN_IDX / $NUM_RUNS  (seed=$SEED)"
    echo "  Model      : $(basename $MODEL_PATH)"
    echo "  Model Type : $MODEL_TYPE"
    echo "  Output dir : $RUN_DIR/evalplus_results/"
    echo "========================================================================"

    CMD="python $WRAPPER_SCRIPT \
        --seed $SEED \
        --model $MODEL_PATH \
        --dataset $DATASET \
        --backend vllm \
        --temperature 0.2 \
        $TYPE_ARGS"

    echo "Executing: $CMD"
    echo ""

    # evalplus writes results to evalplus_results/ under the current directory
    cd "$RUN_DIR"

    START_TIME=$(date +%s)
    eval $CMD
    END_TIME=$(date +%s)

    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    echo ""
    echo "EvalPlus ${DATASET}+ (Run $RUN_IDX) completed in ${MINUTES}m ${SECONDS}s"
    echo "Results saved to: $RUN_DIR/evalplus_results/"
    echo "========================================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# compute_stats DATASET
# ---------------------------------------------------------------------------
compute_stats() {
    local DATASET=$1

    echo "Computing stats for: ${DATASET}+"

    local RESULT_FILES=()
    for i in $(seq 1 $NUM_RUNS); do
        for F in "$BASE_OUTPUT_DIR/run_${i}/evalplus_results/${DATASET}/"*_eval_results.json; do
            if [ -f "$F" ]; then
                RESULT_FILES+=("$F")
            else
                echo "  Missing results for run $i (${DATASET})"
            fi
        done
    done

    if [ ${#RESULT_FILES[@]} -eq 0 ]; then
        echo "  No results found for ${DATASET}, skipping stats."
        return
    fi

    local SUMMARY_DIR="$BASE_OUTPUT_DIR/summary"
    mkdir -p "$SUMMARY_DIR"

    python3 - "${RESULT_FILES[@]}" "$DATASET" "$SUMMARY_DIR" <<'PYEOF'
import sys, json, math, os

result_files = sys.argv[1:-2]
dataset      = sys.argv[-2]
summary_dir  = sys.argv[-1]

def compute_pass1(fpath):
    with open(fpath) as f:
        data = json.load(f)
    evals = data.get("eval", {})
    base_pass = sum(1 for v in evals.values() if v[0]["base_status"] == "pass")
    plus_pass = sum(1 for v in evals.values() if v[0]["plus_status"] == "pass")
    n = len(evals)
    return {
        "base_pass@1": base_pass / n if n else 0.0,
        "plus_pass@1": plus_pass / n if n else 0.0,
    }

all_metrics = {}
for fpath in result_files:
    for k, v in compute_pass1(fpath).items():
        all_metrics.setdefault(k, []).append(v)

def mean_std(vals):
    n  = len(vals)
    mu = sum(vals) / n
    std = math.sqrt(sum((v - mu)**2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return round(mu, 6), round(std, 6)

stats = {}
for key, vals in sorted(all_metrics.items()):
    mu, std = mean_std(vals)
    stats[key] = {"values": vals, "mean": mu, "std": std, "n_runs": len(vals)}

output = {"dataset": dataset, "n_runs": len(result_files), "results": stats}

out_path = os.path.join(summary_dir, f"{dataset}_stats.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"  Stats saved to {out_path}")
print(f"\n  {'Metric':<20} {'Mean':>8}  {'Std':>8}  Values")
print(f"  {'-'*20} {'-'*8}  {'-'*8}  {'-'*30}")
for key, s in stats.items():
    vals_str = "  ".join(f"{v:.4f}" for v in s["values"])
    print(f"  {key:<20} {s['mean']:>8.4f}  {s['std']:>8.4f}  [{vals_str}]")
PYEOF
}

# ---------------------------------------------------------------------------
# run_model — run all tasks for a single model and compute stats
# ---------------------------------------------------------------------------
run_model() {
    echo ""
    echo "###################################################################"
    echo "###  MODEL: $(basename $MODEL_PATH)"
    echo "###  TYPE:  $MODEL_TYPE"
    echo "###  OUTPUT: $BASE_OUTPUT_DIR"
    echo "###################################################################"
    echo ""

    mkdir -p "$BASE_OUTPUT_DIR"

    for RUN_IDX in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "--- RUN $RUN_IDX / $NUM_RUNS ---"
        echo ""

        for DATASET in "${TASK_CONFIGS[@]}"; do
            run_evalplus "$DATASET" "$RUN_IDX"
        done
    done

    echo ""
    echo "--- Computing Mean and Std Dev for $(basename $MODEL_PATH) ---"
    echo ""

    for DATASET in "${TASK_CONFIGS[@]}"; do
        compute_stats "$DATASET"
        echo ""
    done

    echo "Summary for $(basename $MODEL_PATH) (Mean +/- Std):"
    for F in "$BASE_OUTPUT_DIR/summary/"*_stats.json; do
        [ -f "$F" ] || continue
        echo "--- $(basename $F _stats.json) ---"
        python3 -c "
import json
with open('$F') as f:
    d = json.load(f)
for k, s in d['results'].items():
    print(f\"  {k:<20} {s['mean']:.4f} +/- {s['std']:.4f}\")
"
    done
}

# ---------------------------------------------------------------------------
# Main: iterate over all models
# ---------------------------------------------------------------------------
TOTAL_START=$(date +%s)

echo "========================================================================"
echo "EvalPlus Batch Evaluation — ${#MODEL_CONFIGS[@]} model(s) x ${NUM_RUNS} runs"
echo "Datasets : ${TASK_CONFIGS[*]}"
echo "========================================================================"

MODEL_IDX=0
for MODEL_CFG in "${MODEL_CONFIGS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))

    IFS='|' read -r MODEL_PATH BASE_OUTPUT_DIR MODEL_TYPE <<< "$MODEL_CFG"
    MODEL_PATH=$(echo "$MODEL_PATH"           | xargs)
    BASE_OUTPUT_DIR=$(echo "$BASE_OUTPUT_DIR" | xargs)
    MODEL_TYPE=$(echo "$MODEL_TYPE"           | xargs)

    if [ "$MODEL_TYPE" = "base" ]; then
        TYPE_ARGS="--force-base-prompt"
    else
        TYPE_ARGS=""
    fi

    echo ""
    echo "###################################################################"
    echo "###  MODEL $MODEL_IDX / ${#MODEL_CONFIGS[@]}"
    echo "###################################################################"

    MODEL_START=$(date +%s)
    run_model
    MODEL_END=$(date +%s)

    MODEL_ELAPSED=$((MODEL_END - MODEL_START))
    echo ""
    echo "Model $(basename $MODEL_PATH) finished in $((MODEL_ELAPSED/60))m $((MODEL_ELAPSED%60))s"
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "========================================================================"
echo "All Models Completed in $((TOTAL_ELAPSED/60))m $((TOTAL_ELAPSED%60))s"
echo "========================================================================"