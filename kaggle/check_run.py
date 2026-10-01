"""Check a DR-LoRA test run: 4-bit load, routing, growth, loss, memory.

Usage:  !python dr-lora/kaggle/check_run.py
"""

import glob
import json
import os
import re

WORK = os.environ.get("WORK", "/kaggle/working")
OUT = os.environ.get("OUT", f"{WORK}/out_drlora_4bit")
LOG = os.environ.get("LOG", f"{WORK}/train_test.log")
N_LAYERS = int(os.environ.get("N_LAYERS", 16))      # OLMoE-1B-7B
MIN_EVENTS = int(os.environ.get("MIN_EVENTS", 3))   # 20 steps, every 5

log = open(LOG).read()
all_ok = True


def check(name, passed, detail=""):
    global all_ok
    all_ok = all_ok and passed
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")


check("loaded in 4-bit", "use_qlora=True" in log)
check("finished without error",
      "Traceback" not in log and "Training finished" in log)

gate = re.findall(r"\[GateHook\] logits shape=torch.Size\(\[(\d+), (\d+)\]\)",
                  log)
check("routing recorded", bool(gate),
      f"(router output tokens x experts: {gate[0] if gate else None})")

grows = re.findall(r"\[AdaLoRA-Grow\] layer \d+: grew \d+ ranks", log)
events = len(grows) // N_LAYERS
check("ranks grew", events >= MIN_EVENTS,
      f"({events} growth events x {N_LAYERS} layers)")

losses = [float(x) for x in
          re.findall(r"Step: \d+, Loss: ([0-9.eE+-]+|nan|inf)", log)]
check("loss finite", bool(losses) and all(x == x and x < 1e4 for x in losses),
      f"{losses}")
check("LoRA adapter saved",
      os.path.exists(f"{OUT}/adapter_model.safetensors"))

logs = sorted(glob.glob(f"{OUT}/adalora_rank_logs/*.json"))
if logs:
    snap = json.load(open(logs[-1]))
    ranks = [m["active_rank"] for m in snap["per_module_rank"].values()]
    print(f"\nrank log {os.path.basename(logs[-1])}: {len(ranks)} modules, "
          f"mean {sum(ranks) / len(ranks):.2f}, min {min(ranks)}, "
          f"max {max(ranks)}, above 8: {sum(r > 8 for r in ranks)}")

mem_file = f"{WORK}/gpu_mem.csv"
if os.path.exists(mem_file):
    mem = [int(x) for x in open(mem_file).read().split() if x.isdigit()]
    if mem:
        check("GPU memory < 15 GB", max(mem) < 15000,
              f"(peak {max(mem) / 1024:.1f} GB)")

speed = re.findall(r"(\d+\.\d+)s/it", log)
if speed:
    sec = float(speed[-1])
    print(f"time per step ~{sec:.1f} s -> 3,750 steps ~{sec * 3750 / 3600:.1f} h "
          f"(batch 4)")
print("\nALL CHECKS PASSED" if all_ok
      else "\nSOME CHECKS FAILED: send train_test.log")
