"""
evaluate_wmt_translation.py
Translation inference and COMET evaluation script.

Supports:
  - Batch evaluation across multiple models
  - Multiple runs (temperature sampling + random seed) with mean and std dev
  - HF / vLLM inference (configurable per model)
  - Chat template toggle (configurable per model)

Usage:
    python evaluate_wmt_translation.py \
        --test_jsonl data/test.jsonl \
        --output_dir results \
        --cuda_visible_devices 0
"""

import os
import re
import json
import math
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from comet import download_model, load_from_checkpoint
from datasets import load_dataset

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_LPS          = ["en-cs", "en-de", "en-zh"]
DEFAULT_COMET_MODEL = "Unbabel/wmt22-comet-da"
WMT_DATASET_NAME    = "RicardoRei/wmt-da-human-evaluation"
TEST_YEARS          = {2023}

SRC_PATTERN = re.compile(r"Source:\s*(.+?)\nTranslation:", re.DOTALL)

# ══════════════════════════════════════════════════════════════════════════════
# Model configuration
#
# Fields:
#   model_path        : path to model or HF Hub ID
#   output_dir        : output directory for this model
#   use_vllm          : True = vLLM inference, False = HuggingFace inference
#   use_chat_template : True = apply chat template (SFT models),
#                       False = raw prompt (base models)
# ══════════════════════════════════════════════════════════════════════════════
MODEL_CONFIGS = [
    {
        "model_path":        "/path/to/pretrained_models/OLMoE-1B-7B-0924",
        "output_dir":        "/path/to/results/olmoe-1b-7b-0924/base/wmt",
        "use_vllm":          True,
        "use_chat_template": False,
    },
    {
        "model_path":        "/path/to/output/olmoe_wmt_lora_baseline_r32/step_3750/merged_model",
        "output_dir":        "/path/to/results/olmoe-1b-7b-0924/olmoe_wmt_lora_baseline_r32/step_3750",
        "use_vllm":          True,
        "use_chat_template": True,
    },
    {
        "model_path":        "/path/to/output/olmoe_wmt_drlora_r8_target16/step_3750/merged_model",
        "output_dir":        "/path/to/results/olmoe-1b-7b-0924/olmoe_wmt_drlora_r8_target16/step_3750",
        "use_vllm":          True,
        "use_chat_template": True,
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# Multi-run configuration
# ══════════════════════════════════════════════════════════════════════════════
NUM_RUNS    = 3
RUN_SEEDS   = [42, 123, 456]
TEMPERATURE = 0.2       # >0 enables sampling; 0 falls back to greedy decoding


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="WMT translation inference and COMET batch evaluation")
    p.add_argument("--test_jsonl",   default="/path/to/data/WMT/test.jsonl")
    p.add_argument("--comet_model",  default=DEFAULT_COMET_MODEL)
    p.add_argument("--comet_gpus",   type=int, default=0)
    p.add_argument("--comet_batch_size", type=int, default=16)
    p.add_argument("--batch_size",   type=int, default=8,
                   help="HF inference batch size (ignored in vLLM mode).")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--lp",           default=None,
                   help="Evaluate only this language pair (default: all).")
    p.add_argument("--cuda_visible_devices", default=None,
                   help="GPU index(es) to use, e.g. '0' or '1,2'.")
    p.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--hf_cache_dir", default=None)
    p.add_argument("--num_runs",     type=int, default=NUM_RUNS,
                   help=f"Number of runs per model (default: {NUM_RUNS}).")
    p.add_argument("--temperature",  type=float, default=TEMPERATURE,
                   help=f"Sampling temperature; 0 = greedy (default: {TEMPERATURE}).")
    return p.parse_args()


# ── Load test.jsonl ───────────────────────────────────────────────────────────
def load_test_jsonl(path: str, lp_filter: str | None) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            lp = obj.get("subset", "")
            if lp_filter and lp != lp_filter:
                continue
            user_content = obj["messages"][0]["content"]
            m = SRC_PATTERN.search(user_content)
            src = m.group(1).strip() if m else ""
            records.append({
                "lp":       lp,
                "src":      src,
                "prompt":   user_content,
                "messages": obj["messages"],
                "domain":   obj.get("source", ""),
            })
    df = pd.DataFrame(records)
    print(f"Loaded test set: {len(df):,} samples")
    print(df.groupby("lp").size().to_string())
    return df


# ── Load references and human scores from the original dataset ────────────────
def load_references(lps: list[str], hf_cache_dir: str | None) -> pd.DataFrame:
    print(f"\nLoading references and human scores ({WMT_DATASET_NAME}, years {TEST_YEARS})...")
    load_kwargs = {"split": "train"}
    if hf_cache_dir:
        load_kwargs["cache_dir"] = hf_cache_dir
    ds = load_dataset(WMT_DATASET_NAME, **load_kwargs)
    ds = ds.filter(lambda x: int(x["year"]) in TEST_YEARS and x["lp"] in lps)
    ref_df = ds.to_pandas()[["lp", "src", "ref", "score"]].copy()
    ref_df = ref_df.drop_duplicates(subset=["lp", "src"]).reset_index(drop=True)
    print(f"References loaded: {len(ref_df):,} samples")
    return ref_df


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


# ── HuggingFace inference ─────────────────────────────────────────────────────
class HFTranslationModel:
    def __init__(self, model_path: str, max_new_tokens: int):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"\nLoading translation model (HF): {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.device = next(self.model.parameters()).device
        print(f"Model loaded on device: {self.device}")

    def translate(self, prompts: list[str], batch_size: int,
                  temperature: float, seed: int) -> list[str]:
        import torch
        torch.manual_seed(seed)
        do_sample = temperature > 0
        all_translations = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="  Translating", unit="batch"):
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
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )
            all_translations.extend([t.split("\n")[0].strip() for t in decoded])
        return all_translations

    def unload(self):
        import torch, gc
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ── vLLM inference ────────────────────────────────────────────────────────────
class VLLMTranslationModel:
    def __init__(self, model_path: str, max_new_tokens: int,
                 tensor_parallel_size: int, gpu_memory_utilization: float):
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError("Please install vLLM first: pip install vllm")
        print(f"\nLoading translation model (vLLM): {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
        )
        self.max_new_tokens = max_new_tokens
        print("vLLM model loaded.")

    def translate(self, prompts: list[str], batch_size: int = None,
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


# ── Single run: translate and save ───────────────────────────────────────────
def run_translation(
    df: pd.DataFrame,
    trans_model,
    tokenizer,
    lps_to_eval: list[str],
    use_chat_template: bool,
    batch_size: int,
    temperature: float,
    seed: int,
    run_output_dir: str,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    translated_path = os.path.join(run_output_dir, "translated.jsonl")
    if os.path.exists(translated_path):
        print(f"  Translations already exist, skipping: {translated_path}")
        trans_records = [json.loads(l) for l in open(translated_path)]
        df = df.copy()
        df["translated_mt"] = [r["translated_mt"] for r in trans_records]
        return df

    all_mt = []
    for lp in lps_to_eval:
        df_lp = df[df["lp"] == lp].reset_index(drop=True)
        if len(df_lp) == 0:
            continue
        print(f"\n  [{lp}] Translating {len(df_lp):,} samples (seed={seed}, temp={temperature})...")
        prompts = build_inputs(df_lp, tokenizer, use_chat_template)
        translations = trans_model.translate(prompts, batch_size, temperature, seed)
        all_mt.extend(translations)

    df = df[df["lp"].isin(lps_to_eval)].copy().reset_index(drop=True)
    df["translated_mt"] = all_mt

    os.makedirs(run_output_dir, exist_ok=True)
    with open(translated_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"lp": row["lp"], "src": row["src"],
                                "translated_mt": row["translated_mt"]},
                               ensure_ascii=False) + "\n")
    return df


# ── Single run: COMET scoring ─────────────────────────────────────────────────
def run_comet(
    df: pd.DataFrame,
    comet_model,
    lps_to_eval: list[str],
    comet_batch_size: int,
    comet_gpus: int,
    run_output_dir: str,
) -> dict:
    """Returns {lp: system_comet} dict."""
    lp_scores = {}
    all_rows = []

    for lp in lps_to_eval:
        df_lp = df[df["lp"] == lp].reset_index(drop=True)
        if len(df_lp) == 0:
            continue

        data = [
            {"src": row["src"], "mt": row["translated_mt"], "ref": row["ref"]}
            for _, row in df_lp.iterrows()
        ]
        output = comet_model.predict(data, batch_size=comet_batch_size, gpus=comet_gpus)
        df_lp = df_lp.copy()
        df_lp["comet_score"] = output.scores
        lp_scores[lp] = output.system_score

        human_scores = df_lp["score"].tolist()
        pearson_r,  _ = pearsonr(human_scores,  output.scores)
        spearman_r, _ = spearmanr(human_scores, output.scores)
        print(f"    [{lp}] system_comet={output.system_score:.4f}  "
              f"pearson={pearson_r:.4f}  spearman={spearman_r:.4f}")

        all_rows.append(df_lp)

    if all_rows:
        result_df = pd.concat(all_rows, ignore_index=True)
        result_df.to_csv(os.path.join(run_output_dir, "sentence_results.csv"), index=False)

    return lp_scores


# ── Aggregate stats across runs ───────────────────────────────────────────────
def compute_stats(all_run_scores: list[dict], lps: list[str]) -> dict:
    """
    all_run_scores: [{lp: system_comet}, ...] one dict per run.
    Returns {lp: {mean, std, values}}.
    """
    stats = {}
    for lp in lps:
        values = [r[lp] for r in all_run_scores if lp in r]
        if not values:
            continue
        n = len(values)
        mean = sum(values) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
        stats[lp] = {"mean": round(mean, 6), "std": round(std, 6),
                     "values": [round(v, 6) for v in values], "n_runs": n}
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")

    # Disable vLLM v1 async scheduling to avoid EngineCore deadlock
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    lps_to_eval = [args.lp] if args.lp else TARGET_LPS
    seeds = RUN_SEEDS[:args.num_runs]

    # Load test set and references (shared across all models)
    df_base = load_test_jsonl(args.test_jsonl, args.lp)
    df_base = df_base[df_base["lp"].isin(lps_to_eval)].reset_index(drop=True)
    ref_df = load_references(lps_to_eval, args.hf_cache_dir)
    df_base = df_base.merge(ref_df, on=["lp", "src"], how="inner")
    print(f"After merging references: {len(df_base):,} samples")

    # Load COMET model (shared across all models)
    print(f"\nLoading COMET model: {args.comet_model}")
    comet_path  = download_model(args.comet_model)
    comet_model = load_from_checkpoint(comet_path)

    total_start = __import__("time").time()

    for model_idx, cfg in enumerate(MODEL_CONFIGS, 1):
        model_path        = cfg["model_path"]
        model_output_dir  = cfg["output_dir"]
        use_vllm          = cfg["use_vllm"]
        use_chat_template = cfg["use_chat_template"]

        print(f"\n{'#'*67}")
        print(f"###  MODEL {model_idx}/{len(MODEL_CONFIGS)}: {os.path.basename(model_path)}")
        print(f"###  Backend: {'vLLM' if use_vllm else 'HuggingFace'}")
        print(f"###  Chat template: {'enabled' if use_chat_template else 'disabled (pretrain)'}")
        print(f"###  Runs: {args.num_runs}  Temperature: {args.temperature}")
        print(f"{'#'*67}")

        os.makedirs(model_output_dir, exist_ok=True)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if use_vllm:
            trans_model = VLLMTranslationModel(
                model_path, args.max_new_tokens,
                args.vllm_tensor_parallel_size,
                args.vllm_gpu_memory_utilization,
            )
        else:
            trans_model = HFTranslationModel(model_path, args.max_new_tokens)

        all_run_scores = []
        for run_idx, seed in enumerate(seeds, 1):
            print(f"\n--- RUN {run_idx}/{args.num_runs}  (seed={seed}) ---")
            run_output_dir = os.path.join(model_output_dir, f"run_{run_idx}")
            os.makedirs(run_output_dir, exist_ok=True)

            df_run = run_translation(
                df_base.copy(), trans_model, tokenizer, lps_to_eval,
                use_chat_template, args.batch_size, args.temperature,
                seed, run_output_dir,
            )
            lp_scores = run_comet(
                df_run, comet_model, lps_to_eval,
                args.comet_batch_size, args.comet_gpus, run_output_dir,
            )
            all_run_scores.append(lp_scores)

        trans_model.unload()

        stats = compute_stats(all_run_scores, lps_to_eval)

        summary = {"model": model_path, "n_runs": args.num_runs, "results": stats}
        summary_path = os.path.join(model_output_dir, "summary", "wmt_stats.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*55}")
        print(f"  Summary: {os.path.basename(model_path)}  (Mean +/- Std, {args.num_runs} runs)")
        print(f"{'='*55}")
        print(f"  {'lp':<8} {'Mean':>10}  {'Std':>10}  Values")
        for lp, s in stats.items():
            vals_str = "  ".join(f"{v:.4f}" for v in s["values"])
            print(f"  {lp:<8} {s['mean']:.4f}      {s['std']:.4f}      [{vals_str}]")
        print(f"\n  Stats saved to: {summary_path}")

    elapsed = __import__("time").time() - total_start
    print(f"\n{'='*55}")
    print(f"All done! Total elapsed {int(elapsed//60)}m {int(elapsed%60)}s")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()