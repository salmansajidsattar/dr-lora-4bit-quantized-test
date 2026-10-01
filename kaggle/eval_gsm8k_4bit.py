"""GSM8K evaluation of a 4-bit DR-LoRA run, matching DR-LoRA's own eval.

Same as eval/eval_gsm8k.sh: lm-eval task `gsm8k_cot`, 8-shot, greedy,
chat template on (off for the base model), batch 8. The only change: the
model is the 4-bit base + saved LoRA adapter + trained router weights,
instead of a merged bf16 model.

Usage (Kaggle):
  !python dr-lora/kaggle/eval_gsm8k_4bit.py --limit 50     # timing test
  !python dr-lora/kaggle/eval_gsm8k_4bit.py                # all 1,319
  !python dr-lora/kaggle/eval_gsm8k_4bit.py --no_adapter   # 4-bit base
"""

import argparse
import json
import os
import time

import torch
from peft import PeftModel
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

WORK = os.environ.get("WORK", "/kaggle/working")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="allenai/OLMoE-1B-7B-0924")
    p.add_argument("--run_dir", default=f"{WORK}/out_drlora_4bit_full",
                   help="training output folder (adapter + router)")
    p.add_argument("--no_adapter", action="store_true",
                   help="evaluate the 4-bit base model only")
    p.add_argument("--limit", type=int, default=None,
                   help="evaluate only the first N questions")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--tasks", default="gsm8k_cot")
    p.add_argument("--include_path", default=None)  # for local tests
    p.add_argument("--no_quant", action="store_true",
                   help="CPU tests only: load the base without 4-bit")
    p.add_argument("--out", default=f"{WORK}/eval_gsm8k")
    return p.parse_args()


def load_model(args):
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        device_map = {"": torch.cuda.current_device()}
    else:
        dtype, device_map = torch.float32, {"": "cpu"}

    # Same 4-bit setup as training: routers and lm_head stay full precision.
    n_layers = AutoConfig.from_pretrained(args.base).num_hidden_layers
    keep = ["lm_head"] + [f"model.layers.{i}.mlp.gate"
                          for i in range(n_layers)]
    quant = None if args.no_quant else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        llm_int8_skip_modules=keep)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, device_map=device_map,
        attn_implementation="eager", quantization_config=quant)
    if args.no_adapter:
        return model, AutoTokenizer.from_pretrained(args.base), dtype

    model = PeftModel.from_pretrained(model, args.run_dir)
    router_file = os.path.join(args.run_dir, "router_state_dict.pt")
    if os.path.exists(router_file):
        router = torch.load(router_file, map_location="cpu")
        params = dict(model.named_parameters())
        for name, value in router.items():
            params[name].data.copy_(value.to(params[name].dtype))
        print(f"loaded {len(router)} trained router tensors")
    else:
        print("WARNING: no router_state_dict.pt; using the original router")
    # The run's saved tokenizer has the tulu chat template used in training.
    return model, AutoTokenizer.from_pretrained(args.run_dir), dtype


def main():
    args = parse_args()
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    model, tokenizer, dtype = load_model(args)
    model.eval()
    lm = HFLM(pretrained=model, tokenizer=tokenizer,
              batch_size=args.batch_size)

    start = time.time()
    results = simple_evaluate(
        model=lm, tasks=args.tasks.split(","), num_fewshot=8,
        apply_chat_template=not args.no_adapter, limit=args.limit,
        task_manager=TaskManager(include_path=args.include_path),
        log_samples=True,
    )
    minutes = (time.time() - start) / 60

    name = "base_4bit" if args.no_adapter else "drlora_4bit"
    if args.limit:
        name += f"_limit{args.limit}"
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{name}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    for task, res in results["results"].items():
        n = len({s["doc_id"] for s in results["samples"][task]})
        print(f"\n{task}: {n} questions, {minutes:.1f} min "
              f"({minutes * 60 / max(n, 1):.1f} s/question), dtype {dtype}")
        for key, value in res.items():
            if key.startswith("exact_match,"):
                print(f"  {key.split(',')[1]:<17} {100 * value:.2f}%")
    print(f"\nsaved {args.out}/{name}.json")
    print("Paper (OLMoE-1B-7B, bf16, batch 48): DR-LoRA 28.4, LoRA 25.2")


if __name__ == "__main__":
    main()
