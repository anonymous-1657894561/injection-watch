"""E136 step 1, dev only: the selection grid for the fine-tuned reasoning-prefix classifier. Local GPU, $0.

Grid (seed 0, one run per cell, scored on dev eval-v1.2 rows after every epoch of a 4-epoch schedule):
  config in {smoke, realistic} x N in {16, 32} (the priority N; the frozen setting is applied to 8, 64, 128 unselected) x lr in {1e-5, 2e-5, 5e-5} x class weight in {balanced, none} x epoch 1..4.
One setting (lr, class weight, epoch) is frozen for every (config, N): the one with the highest mean within-case AUROC over the
cells run (as E134 froze one C). Resumable: finished runs are kept in results/e136/dev-grid.jsonl.

  python e136_grid.py [N ...]   -> results/e136/dev-grid.jsonl
"""
import json
import sys

from e136_common import OUT, gpu_lock, load, train
from injecforks_metrics import within_case_mean

LRS, CWS = [1e-5, 2e-5, 5e-5], ["balanced", "none"]
NS_ORDER = [int(a) for a in sys.argv[1:]] or [16, 32]
OUT.mkdir(parents=True, exist_ok=True)
F = OUT / "dev-grid.jsonl"
done = {(x["cfg"], x["N"], x["lr"], x["cw"]) for x in map(json.loads, open(F))} if F.exists() else set()
dev = {c: load(c, "dev") for c in ("smoke", "realistic")}
for N in NS_ORDER:
    for cfg in ("realistic", "smoke"):
        rs = [r for r in dev[cfg] if r["cut"].get(N) is not None]
        for lr in LRS:
            for cw in CWS:
                if (cfg, N, lr, cw) in done:
                    continue
                with gpu_lock():
                    o = train(cfg, N, lr, cw, 0, evals={"dev": rs})
                au = {ep: within_case_mean([r["case"] for r in rs], [r["violates"] for r in rs], [s["dev"][r["row_id"]] for r in rs])
                      for ep, s in o.items()}
                rec = dict(cfg=cfg, N=N, lr=lr, cw=cw, seed=0, auroc={str(k): v for k, v in au.items()})
                print(json.dumps(rec), flush=True)
                with open(F, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
