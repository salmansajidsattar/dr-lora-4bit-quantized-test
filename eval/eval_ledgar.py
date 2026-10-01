"""
evaluate_ledgar_mc.py
Multiple-choice inference and accuracy evaluation on LEDGAR (for base models).

Differences from evaluate_ledgar.py:
  - Reads test_mc.jsonl (four-choice multiple-choice format)
  - Model output is an option letter (A/B/C/D); matching is done by letter
  - Predicted label name is recovered from the options field for analysis

Usage:
    python evaluate_ledgar_mc.py \
        --test_jsonl data/LEDGAR/test_mc.jsonl \
        --cuda_visible_devices 0
"""

import os
import json
import math
import argparse
import random
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_TEST_JSONL = "/path/to/data/LEDGAR/test_mc.jsonl"

MODEL_CONFIGS = [
    {
        "model_path":        "/path/to/pretrained_models/OLMoE-1B-7B-0924",
        "output_dir":        "/path/to/results/olmoe-1b-7b-0924/base/ledgar_mc",
        "use_vllm":          True,
        "use_chat_template": False,
    },
    {
        "model_path":        "/path/to/output/olmoe_ledgar_lora_baseline_r32/step_3750/merged_model",
        "output_dir":        "/path/to/results/olmoe-1b-7b-0924/olmoe_ledgar_lora_baseline_r32/step_3750/ledgar_mc",
        "use_vllm":          True,
        "use_chat_template": False,
    },
]

NUM_RUNS    = 3
RUN_SEEDS   = [42, 123, 456]
TEMPERATURE = 0.2
LETTERS     = ["A", "B", "C", "D"]


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LEDGAR MC inference and accuracy evaluation")
    p.add_argument("--test_jsonl",   default=DEFAULT_TEST_JSONL)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=4,
                   help="Only A/B/C/D needed; 4 tokens is sufficient.")
    p.add_argument("--cuda_visible_devices", default=None)
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--num_runs",     type=int, default=NUM_RUNS)
    p.add_argument("--temperature",  type=float, default=TEMPERATURE)
    return p.parse_args()


# ── Load test_mc.jsonl ────────────────────────────────────────────────────────
def load_test_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "prompt":         obj["messages"][0]["content"],
                "messages":       obj["messages"],
                "label":          obj.get("subset", ""),
                "correct_letter": obj.get("correct_letter", ""),
                "options":        obj.get("options", {}),
            })
    df = pd.DataFrame(records)
    print(f"Loaded test set: {len(df):,} samples  |  labels: {df['label'].nunique()}")
    return df


# ── Build model inputs ────────────────────────────────────────────────────────
def build_inputs(df: pd.DataFrame, tokenizer, use_chat_template: bool) -> list[str]:
    if not use_chat_template:
        return df["prompt"].tolist()
    inputs = []
    for messages in df["messages"]:
        user_messages = [m for m in messages if m["role"] == "user"]
        text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True,
        )
        inputs.append(text)
    return inputs


# ── Answer extraction: extract first A/B/C/D from generated text ──────────────
def extract_letter(generated: str) -> str:
    generated = generated.strip()
    m = re.match(r"^\s*([ABCD])\b", generated, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", generated, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ""


# ── HuggingFace inference ─────────────────────────────────────────────────────
class HFModel:
    def __init__(self, model_path: str, max_new_tokens: int,
                 trust_remote_code: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"\nLoading model (HF): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")

    def generate(self, prompts: list[str], batch_size: int,
                 temperature: float, seed: int) -> list[str]:
        import torch
        torch.manual_seed(seed)
        do_sample = temperature > 0
        all_outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="  Inference", unit="batch"):
            batch = prompts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=1024,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=0.9 if do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )
            all_outputs.extend(decoded)
        return all_outputs

    def unload(self):
        import torch, gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ── vLLM inference ────────────────────────────────────────────────────────────
class VLLMModel:
    def __init__(self, model_path: str, max_new_tokens: int,
                 tensor_parallel_size: int, gpu_memory_utilization: float):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("Please install vLLM first: pip install vllm")
        print(f"\nLoading model (vLLM): {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompts: list[str], batch_size: int = None,
                 temperature: float = 0.2, seed: int = 42) -> list[str]:
        from vllm import SamplingParams
        do_sample = temperature > 0
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=temperature if do_sample else 0,
            seed=seed,
            stop=["\n"],
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [o.outputs[0].text.strip() for o in outputs]

    def unload(self):
        import gc, torch
        del self.llm
        gc.collect()
        torch.cuda.empty_cache()


# ── Single run: inference and save ───────────────────────────────────────────
def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    use_chat_template: bool,
    batch_size: int,
    temperature: float,
    seed: int,
    run_output_dir: str,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    pred_path = os.path.join(run_output_dir, "predictions.jsonl")
    if os.path.exists(pred_path):
        print(f"  Predictions already exist, skipping inference: {pred_path}")
        pred_records = [json.loads(l) for l in open(pred_path)]
        df = df.copy()
        df["pred_letter"] = [r["pred_letter"] for r in pred_records]
        df["pred_label"]  = [r.get("pred_label", "") for r in pred_records]
        return df

    print(f"\n  Running inference on {len(df):,} samples (seed={seed}, temp={temperature})...")
    prompts = build_inputs(df, tokenizer, use_chat_template)
    outputs = model.generate(prompts, batch_size, temperature, seed)

    pred_letters = [extract_letter(o) for o in outputs]
    pred_labels = [
        row["options"].get(letter, "")
        for row, letter in zip(df.to_dict("records"), pred_letters)
    ]

    df = df.copy()
    df["pred_letter"] = pred_letters
    df["pred_label"]  = pred_labels

    os.makedirs(run_output_dir, exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({
                "label":          row["label"],
                "correct_letter": row["correct_letter"],
                "pred_letter":    row["pred_letter"],
                "pred_label":     row["pred_label"],
            }, ensure_ascii=False) + "\n")
    return df


# ── Single run: compute accuracy ─────────────────────────────────────────────
def compute_accuracy(df: pd.DataFrame, run_output_dir: str) -> float:
    correct = (df["pred_letter"] == df["correct_letter"]).sum()
    failed  = (df["pred_letter"] == "").sum()
    accuracy = correct / len(df)

    result_df = df[["label", "correct_letter", "pred_letter", "pred_label"]].copy()
    result_df["correct"] = result_df["pred_letter"] == result_df["correct_letter"]
    result_df.to_csv(os.path.join(run_output_dir, "results.csv"), index=False)

    label_acc = result_df.groupby("label")["correct"].mean().sort_values()
    label_acc.to_csv(os.path.join(run_output_dir, "per_label_accuracy.csv"))

    print(f"    Accuracy={accuracy:.4f}  ({correct}/{len(df)})  "
          f"failed_extract={failed}")
    return accuracy


# ── Aggregate stats across runs ───────────────────────────────────────────────
def compute_stats(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": round(mean, 6), "std": round(std, 6),
            "values": [round(v, 6) for v in values], "n_runs": n}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    seeds = RUN_SEEDS[:args.num_runs]
    df_base = load_test_jsonl(args.test_jsonl)

    total_start = __import__("time").time()

    for model_idx, cfg in enumerate(MODEL_CONFIGS, 1):
        model_path        = cfg["model_path"]
        model_output_dir  = cfg["output_dir"]
        use_vllm          = cfg["use_vllm"]
        use_chat_template = cfg["use_chat_template"]
        trust_remote_code = cfg.get("trust_remote_code", False)

        print(f"\n{'#'*67}")
        print(f"###  MODEL {model_idx}/{len(MODEL_CONFIGS)}: {os.path.basename(model_path)}")
        print(f"###  Backend: {'vLLM' if use_vllm else 'HuggingFace'}")
        print(f"###  Chat template: {'enabled' if use_chat_template else 'disabled (pretrain)'}")
        print(f"###  Runs: {args.num_runs}  Temperature: {args.temperature}")
        print(f"{'#'*67}")

        os.makedirs(model_output_dir, exist_ok=True)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_vllm:
            model = VLLMModel(
                model_path, args.max_new_tokens,
                args.vllm_tensor_parallel_size,
                args.vllm_gpu_memory_utilization,
            )
        else:
            model = HFModel(model_path, args.max_new_tokens, trust_remote_code)

        all_accuracies = []
        for run_idx, seed in enumerate(seeds, 1):
            print(f"\n--- RUN {run_idx}/{args.num_runs}  (seed={seed}) ---")
            run_output_dir = os.path.join(model_output_dir, f"run_{run_idx}")
            os.makedirs(run_output_dir, exist_ok=True)

            df_run = run_inference(
                df_base.copy(), model, tokenizer,
                use_chat_template, args.batch_size,
                args.temperature, seed, run_output_dir,
            )
            accuracy = compute_accuracy(df_run, run_output_dir)
            all_accuracies.append(accuracy)

        model.unload()

        stats = compute_stats(all_accuracies)
        summary = {"model": model_path, "n_runs": args.num_runs, "accuracy": stats}
        summary_path = os.path.join(model_output_dir, "summary", "ledgar_mc_stats.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        vals_str = "  ".join(f"{v:.4f}" for v in stats["values"])
        print(f"\n{'='*55}")
        print(f"  Summary: {os.path.basename(model_path)}  (Mean +/- Std, {args.num_runs} runs)")
        print(f"{'='*55}")
        print(f"  Accuracy : {stats['mean']*100:.2f}% +/- {stats['std']*100:.2f}%  [{vals_str}]")
        print(f"  Note: random-guess baseline is 25.00%")
        print(f"\n  Stats saved to: {summary_path}")

    elapsed = __import__("time").time() - total_start
    print(f"\n{'='*55}")
    print(f"All done! Total elapsed {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()