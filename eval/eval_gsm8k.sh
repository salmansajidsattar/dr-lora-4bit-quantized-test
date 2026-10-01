#!/bin/bash
# ================= Configuration =================

# ===== Model Configuration =====
# Format: one model per line, fields separated by |
#   MODEL_PATH | BASE_OUTPUT_DIR | USE_CHAT_TEMPLATE
#
# USE_CHAT_TEMPLATE: true / false
# Comment/uncomment lines to enable/disable models.
MODEL_CONFIGS=(
    "/path/to/pretrained_models/OLMoE-1B-7B-0924 | /path/to/results/olmoe-1b-7b-0924/base | false"
    "/path/to/output/olmoe_gsm8k_metamathqa_lora_baseline_r16/step_3750/merged_model | /path/to/results/olmoe-1b-7b-0924/lora_baseline_r16/step_3750 | true"
    "/path/to/output/olmoe_gsm8k_metamathqa_drlora_r8_target16/step_3750/merged_model | /path/to/results/olmoe-1b-7b-0924/drlora_r8_target16/step_3750 | true"
)

CUDA_DEVICE="0"

# ===== Multi-run Configuration =====
NUM_RUNS=3

# Random seed for each run
RUN_SEEDS=(42 123 456)

# ===== Quick Test Configuration =====
QUICK_TEST=false  # Set to false for full evaluation
TEST_LIMIT=10     # Number of samples per task in quick test mode

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# ===== MBPP requires code execution permission =====
export HF_ALLOW_CODE_EVAL=1

# ==========================================
# ===== Task Configuration =====
TASK_CONFIGS=(
    "gsm8k_cot | 8 |                            | "
    "ifeval     | 0 |                            | "
)

# ==========================================
# HF backend batch sizes (fixed integers)
declare -A BATCH_SIZES
BATCH_SIZES["gsm8k_cot"]="8"
BATCH_SIZES["ifeval"]="8"
BATCH_SIZES["default"]="4"

# ==========================================
# make_hf_args SEED [with_chat]
#   Uses HuggingFace backend with trust_remote_code support
# ==========================================
make_hf_args() {
    local SEED=$1
    local WITH_CHAT=${2:-""}
    local ARGS="--model hf \
    --model_args pretrained=${MODEL_PATH},trust_remote_code=True,dtype=bfloat16,device_map=auto \
    --log_samples"
    if [ "$WITH_CHAT" = "with_chat" ]; then
        echo "$ARGS --apply_chat_template"
    else
        echo "$ARGS"
    fi
}

# ---------------------------------------------------------------------------
# run_task TASK_NAME FEW_SHOT EXTRA_ARGS CHAT_OVERRIDE RUN_IDX
# ---------------------------------------------------------------------------
run_task() {
    TASK_NAME=$1
    FEW_SHOT=$2
    EXTRA_ARGS=$3
    CHAT_OVERRIDE=${4:-""}
    RUN_IDX=$5

    local SEED=${RUN_SEEDS[$((RUN_IDX - 1))]}

    if [[ -v BATCH_SIZES[$TASK_NAME] ]]; then
        BATCH_SIZE=${BATCH_SIZES[$TASK_NAME]}
    else
        BATCH_SIZE=${BATCH_SIZES["default"]}
    fi

    if [ "$CHAT_OVERRIDE" = "no_chat_template" ]; then
        TASK_HF_ARGS=$(make_hf_args "$SEED")
        echo "Chat template disabled for task: $TASK_NAME"
    elif [ "$USE_CHAT_TEMPLATE" = true ]; then
        TASK_HF_ARGS=$(make_hf_args "$SEED" "with_chat")
    else
        TASK_HF_ARGS=$(make_hf_args "$SEED")
    fi

    if [ "$QUICK_TEST" = true ]; then
        LIMIT_ARG="--limit $TEST_LIMIT"
        echo "QUICK TEST MODE: Testing only $TEST_LIMIT samples"
    else
        LIMIT_ARG=""
        echo "FULL TEST MODE: Testing all samples"
    fi

    echo "================================================================"
    echo "Running Task: $TASK_NAME | Run: $RUN_IDX/$NUM_RUNS | Few-shot: $FEW_SHOT | Batch: $BATCH_SIZE | Seed: $SEED"
    echo "Model: $(basename $MODEL_PATH)"
    echo "================================================================"

    OUTPUT_PATH="$BASE_OUTPUT_DIR/run_${RUN_IDX}/$TASK_NAME"
    mkdir -p "$OUTPUT_PATH"

    CMD="lm_eval $TASK_HF_ARGS --batch_size $BATCH_SIZE --tasks $TASK_NAME --num_fewshot $FEW_SHOT --output_path $OUTPUT_PATH $LIMIT_ARG $EXTRA_ARGS"

    echo "Executing: $CMD"
    echo ""

    START_TIME=$(date +%s)
    eval $CMD
    END_TIME=$(date +%s)

    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))

    echo ""
    echo "Task $TASK_NAME (Run $RUN_IDX) completed in ${MINUTES}m ${SECONDS}s"
    echo "================================================================"
    echo ""
}

# ---------------------------------------------------------------------------
# compute_stats TASK_NAME
# ---------------------------------------------------------------------------
compute_stats() {
    TASK_NAME=$1

    echo "Computing stats for: $TASK_NAME"

    RESULT_FILES=()
    for i in $(seq 1 $NUM_RUNS); do
        FOUND=$(find "$BASE_OUTPUT_DIR/run_${i}/$TASK_NAME" -name "results_*.json" 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            RESULT_FILES+=("$FOUND")
        else
            echo "  Missing results for run $i ($TASK_NAME)"
        fi
    done

    if [ ${#RESULT_FILES[@]} -eq 0 ]; then
        echo "  No results found for $TASK_NAME, skipping stats."
        return
    fi

    SUMMARY_DIR="$BASE_OUTPUT_DIR/summary"
    mkdir -p "$SUMMARY_DIR"

    python3 - "${RESULT_FILES[@]}" "$TASK_NAME" "$SUMMARY_DIR" <<'PYEOF'
import sys, json, math, os

result_files = sys.argv[1:-2]
task_name    = sys.argv[-2]
summary_dir  = sys.argv[-1]

all_metrics = {}

for fpath in result_files:
    with open(fpath) as f:
        data = json.load(f)
    task_results = data.get("results", {})
    for tname, metrics in task_results.items():
        for metric_key, val in metrics.items():
            if isinstance(val, (int, float)) and not metric_key.endswith("_stderr"):
                full_key = f"{tname}/{metric_key}"
                all_metrics.setdefault(full_key, []).append(float(val))

stats = {}
for key, vals in sorted(all_metrics.items()):
    n   = len(vals)
    mu  = sum(vals) / n
    if n > 1:
        variance = sum((v - mu) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    stats[key] = {"values": vals, "mean": round(mu, 6), "std": round(std, 6), "n_runs": n}

output = {"task": task_name, "n_runs": len(result_files), "results": stats}

out_path = os.path.join(summary_dir, f"{task_name}_stats.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"  Stats saved to {out_path}")
print(f"\n  {'Metric':<55} {'Mean':>10}  {'Std':>10}  {'Values'}")
print(f"  {'-'*55} {'-'*10}  {'-'*10}  {'-'*30}")
for key, s in stats.items():
    vals_str = "  ".join(f"{v:.4f}" for v in s["values"])
    print(f"  {key:<55} {s['mean']:>10.4f}  {s['std']:>10.4f}  [{vals_str}]")
PYEOF
}

# ---------------------------------------------------------------------------
# run_model  —  run all tasks for a single model and compute stats
# ---------------------------------------------------------------------------
run_model() {
    echo ""
    echo "###################################################################"
    echo "###  MODEL: $(basename $MODEL_PATH)"
    echo "###  OUTPUT: $BASE_OUTPUT_DIR"
    echo "###  CHAT TEMPLATE: $USE_CHAT_TEMPLATE"
    echo "###################################################################"
    echo ""

    mkdir -p "$BASE_OUTPUT_DIR"

    for RUN_IDX in $(seq 1 $NUM_RUNS); do
        echo ""
        echo "--- RUN $RUN_IDX / $NUM_RUNS ---"
        echo ""

        for CFG in "${TASK_CONFIGS[@]}"; do
            IFS='|' read -r T_NAME T_FEWSHOT T_EXTRA T_CHAT <<< "$CFG"
            T_NAME=$(echo "$T_NAME"       | xargs)
            T_FEWSHOT=$(echo "$T_FEWSHOT" | xargs)
            T_EXTRA=$(echo "$T_EXTRA"     | xargs)
            T_CHAT=$(echo "$T_CHAT"       | xargs)
            run_task "$T_NAME" "$T_FEWSHOT" "$T_EXTRA" "$T_CHAT" "$RUN_IDX"
        done
    done

    echo ""
    echo "--- Computing Mean and Std Dev for $(basename $MODEL_PATH) ---"
    echo ""

    for CFG in "${TASK_CONFIGS[@]}"; do
        IFS='|' read -r T_NAME _ _ _ <<< "$CFG"
        T_NAME=$(echo "$T_NAME" | xargs)
        compute_stats "$T_NAME"
        echo ""
    done

    echo "Summary for $(basename $MODEL_PATH) (Mean +/- Std):"
    for F in "$BASE_OUTPUT_DIR/summary/"*_stats.json; do
        [ -f "$F" ] || continue
        echo "--- $(basename $F .json) ---"
        python3 -c "
import json
with open('$F') as f:
    d = json.load(f)
for k, s in d['results'].items():
    print(f\"  {k:<55} {s['mean']:.4f} +/- {s['std']:.4f}\")
"
    done
}

# ---------------------------------------------------------------------------
# Main: iterate over all models
# ---------------------------------------------------------------------------
TOTAL_START=$(date +%s)

echo "========================================================================"
echo "LM-Eval Batch Evaluation (HF backend) — ${#MODEL_CONFIGS[@]} model(s) x ${NUM_RUNS} runs"
echo "Quick Test: $QUICK_TEST (Limit: $TEST_LIMIT samples per task)"
echo "CUDA_DEVICE: $CUDA_DEVICE"
echo "========================================================================"

MODEL_IDX=0
for MODEL_CFG in "${MODEL_CONFIGS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))

    IFS='|' read -r MODEL_PATH BASE_OUTPUT_DIR USE_CHAT_TEMPLATE <<< "$MODEL_CFG"
    MODEL_PATH=$(echo "$MODEL_PATH"               | xargs)
    BASE_OUTPUT_DIR=$(echo "$BASE_OUTPUT_DIR"     | xargs)
    USE_CHAT_TEMPLATE=$(echo "$USE_CHAT_TEMPLATE" | xargs)

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