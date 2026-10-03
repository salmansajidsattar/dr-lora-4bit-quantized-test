"""Combine the per-GPU GSM8K results from eval_2gpu.sh into one score.

Usage: python merge_eval.py part0.json part1.json
"""

import json
import sys

files = sys.argv[1:]
parts = [json.load(open(f)) for f in files]

merged = {}
for task in parts[0]["results"]:
    total = 0
    sums = {}
    for res in parts:
        n = len({s["doc_id"] for s in res["samples"][task]})
        total += n
        for key, value in res["results"][task].items():
            if key.startswith("exact_match,"):
                sums[key] = sums.get(key, 0) + value * n
    print(f"\n{task}: {total} questions (from {len(files)} parts)")
    merged[task] = {"questions": total}
    for key, value in sums.items():
        merged[task][key] = value / total
        print(f"  {key.split(',')[1]:<17} {100 * value / total:.2f}%")

out = files[0].replace("_part0of2", "_merged")
with open(out, "w") as f:
    json.dump({"files": files, "results": merged}, f, indent=2)
print(f"\nsaved {out}")
print("Paper (OLMoE-1B-7B, bf16, batch 48): DR-LoRA 28.4, LoRA 25.2")
