#!/usr/bin/env python3
"""
eval_medmcqa.py
Generative evaluation on MedMCQA dev/test set using vLLM.

Evaluation approach:
  - Present each question with four options and prompt the model to output a letter (A/B/C/D).
  - Extract the first option letter from the model output and compare with the ground truth.
  - Report accuracy separately for single/multi question types and overall.

Usage:
  # Evaluate a fine-tuned model
  python eval_medmcqa.py \
      --model_path /path/to/merged_model \
      --data_path  ./MedMCQA/dev.json \
      --output_path ./results/my_model/step_900

  # Evaluate a base model
  python eval_medmcqa.py \
      --model_path /path/to/pretrained_models/OLMoE-1B-7B-0924 \
      --data_path  ./MedMCQA/dev.json \
      --output_path ./results/base

  # Specify random seed (passed by shell script for multi-run evaluation)
  python eval_medmcqa.py \
      --model_path /path/to/model \
      --data_path  ./MedMCQA/dev.json \
      --output_path ./results/run_1 \
      --seed 42

  # Quick debug (first 50 samples only)
  python eval_medmcqa.py \
      --model_path /path/to/model \
      --data_path  ./MedMCQA/dev.json \
      --output_path ./results/debug \
      --max_samples 50
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    from vllm import LLM, SamplingParams
except ImportError:
    sys.exit("[Error] Please install vllm first: pip install vllm")

# cop integer -> option letter
COP_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}
# option letter -> field name
LETTER_TO_KEY = {"A": "opa", "B": "opb", "C": "opc", "D": "opd"}


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(example: dict) -> str:
    """
    Build a 0-shot prompt asking the model to output only the option letter.
    Format:
        Question: <question>

        A. <opa>
        B. <opb>
        C. <opc>
        D. <opd>

        Answer with only the letter of the correct option (A, B, C, or D).
        Answer:
    """
    q   = example.get("question", "").strip()
    opa = example.get("opa", "").strip()
    opb = example.get("opb", "").strip()
    opc = example.get("opc", "").strip()
    opd = example.get("opd", "").strip()

    return (
        f"Question: {q}\n\n"
        f"A. {opa}\n"
        f"B. {opb}\n"
        f"C. {opc}\n"
        f"D. {opd}\n\n"
        f"Answer with only the letter of the correct option (A, B, C, or D).\n"
        f"Answer:"
    )


# ── Answer extraction ─────────────────────────────────────────────────────────

def extract_answer(output: str) -> str | None:
    """
    Extract the first option letter (A/B/C/D) from the model output.
    Prefers a letter at the start of the string; falls back to the first
    standalone letter found anywhere in the output.
    Returns an uppercase letter or None if extraction fails.
    """
    output = output.strip()
    m = re.match(r"^\s*([ABCD])\b", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


# ── Main evaluation logic ─────────────────────────────────────────────────────

def evaluate(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"[1/4] Loading data: {args.data_path}")
    examples = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ex.get("cop") not in COP_TO_LETTER:
                continue
            examples.append(ex)

    if args.max_samples:
        examples = examples[: args.max_samples]

    print(f"      Valid samples: {len(examples)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[2/4] Loading model: {args.model_path}")
    print(f"      Seed: {args.seed}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        max_model_len=512,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.2,
        top_p=1.0,
        max_tokens=16,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"[3/4] Generating ({len(examples)} samples)...")
    prompts = [build_prompt(ex) for ex in examples]

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"      Generation complete, elapsed {elapsed:.1f}s")

    # ── Score ─────────────────────────────────────────────────────────────────
    print(f"[4/4] Computing accuracy...")

    results = []
    correct_total = 0
    correct_single = 0
    correct_multi  = 0
    total_single   = 0
    total_multi    = 0
    failed         = 0

    for ex, out in zip(examples, outputs):
        generated = out.outputs[0].text
        pred      = extract_answer(generated)
        gold      = COP_TO_LETTER[ex["cop"]]
        is_correct = (pred == gold)

        choice_type = ex.get("choice_type", "single")
        if choice_type == "single":
            total_single  += 1
            correct_single += int(is_correct)
        else:
            total_multi  += 1
            correct_multi += int(is_correct)

        correct_total += int(is_correct)
        if pred is None:
            failed += 1

        results.append({
            "id":          ex.get("id", ""),
            "question":    ex.get("question", ""),
            "gold":        gold,
            "pred":        pred,
            "correct":     is_correct,
            "choice_type": choice_type,
            "generated":   generated.strip(),
        })

    n = len(examples)
    acc_total  = correct_total  / n             if n             else 0.0
    acc_single = correct_single / total_single  if total_single  else 0.0
    acc_multi  = correct_multi  / total_multi   if total_multi   else 0.0

    def stderr(correct: int, total: int) -> float:
        """
        Binomial standard error: sqrt(p * (1-p) / n).
        Consistent with lm-eval's computation for single-run greedy evaluation.
        """
        if total == 0:
            return 0.0
        p = correct / total
        return (p * (1 - p) / total) ** 0.5

    se_total  = stderr(correct_total,  n)
    se_single = stderr(correct_single, total_single)
    se_multi  = stderr(correct_multi,  total_multi)

    # ── Print results ─────────────────────────────────────────────────────────
    summary = {
        "model":        args.model_path,
        "data":         args.data_path,
        "seed":         args.seed,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total":        n,
        "accuracy": {
            "overall": round(acc_total,  4),
            "single":  round(acc_single, 4),
            "multi":   round(acc_multi,  4),
        },
        "stderr": {
            "overall": round(se_total,  4),
            "single":  round(se_single, 4),
            "multi":   round(se_multi,  4),
        },
        "counts": {
            "correct_total":  correct_total,
            "total_single":   total_single,
            "correct_single": correct_single,
            "total_multi":    total_multi,
            "correct_multi":  correct_multi,
            "failed_extract": failed,
        },
        "elapsed_sec": round(elapsed, 1),
    }

    print("\n" + "=" * 60)
    print("MedMCQA Evaluation Results")
    print("=" * 60)
    print(f"  Model   : {os.path.basename(args.model_path)}")
    print(f"  Seed    : {args.seed}")
    print(f"  Samples : {n}")
    print(f"  Overall Accuracy : {acc_total*100:.2f}% +/- {se_total*100:.2f}%  ({correct_total}/{n})")
    print(f"  Single  Accuracy : {acc_single*100:.2f}% +/- {se_single*100:.2f}%  ({correct_single}/{total_single})")
    print(f"  Multi   Accuracy : {acc_multi*100:.2f}% +/- {se_multi*100:.2f}%  ({correct_multi}/{total_multi})")
    print(f"  Failed Extract   : {failed}")
    print("=" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)

    summary_file = os.path.join(args.output_path, "results_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    samples_file = os.path.join(args.output_path, "results_samples.jsonl")
    with open(samples_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  Results saved:")
    print(f"    Summary : {summary_file}")
    print(f"    Samples : {samples_file}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on MedMCQA dev/test set")
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model directory (e.g. merged_model).",
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the MedMCQA dev/test json file.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Directory to write evaluation results.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum number of samples to evaluate (default: all).",
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="GPU index to use, e.g. '0' (default: 0).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed; passed by shell script for multi-run evaluation (default: 42).",
    )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    evaluate(args)


if __name__ == "__main__":
    main()