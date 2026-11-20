"""Repack E79 v2 injected-row states (N>0) into per-layer float16 arrays on native fs, plus an index jsonl."""
import json, sys, numpy as np
from pathlib import Path
VF = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(VF))
from injecforks_load import load
OUT = Path(__file__).resolve().parent.parent / "data/e144-states"; OUT.mkdir(exist_ok=True)
D = VF / "data/e79v2/states-train/user-e79v2-9b-pod-20261026T122152Z/states"
keep = {}
for cfg in ("realistic", "smoke"):
    for sp in ("train", "dev", "test"):
        for r in (load(cfg, sp, subset=None) if sp == "train" else load(cfg, sp)):
            keep[r["row_id"]] = (cfg, sp, r["case"], int(bool(r["violates"])))
files = sorted(D.glob("states-*.npy"))
sel = []
for f in files:
    meta = [json.loads(l) for l in open(f.with_name(f.name.replace("states-", "meta-").replace(".npy", ".jsonl")))]
    sel.append([(j, m) for j, m in enumerate(meta) if m["kind"] == "prefix" and m["row_id"] in keep])
tot = sum(map(len, sel)); L, d = np.load(files[0], mmap_mode="r").shape[1:]
print("rows", tot, "layers", L, "d", d, flush=True)
mm = [np.lib.format.open_memmap(OUT / f"L{l}.npy", "w+", np.float16, (tot, d)) for l in range(L)]
idx, k = open(OUT / "index.jsonl", "w"), 0
for f, s in zip(files, sel):
    X = np.load(f)  # sequential read
    js = [j for j, _ in s]
    for l in range(L):
        mm[l][k:k + len(js)] = X[js, l]
    for j, m in s:
        cfg, sp, case, y = keep[m["row_id"]]
        idx.write(json.dumps(dict(row_id=m["row_id"], N=m["N"], cfg=cfg, split=sp, case=case, y=y)) + "\n")
    k += len(js); print(f.name, k, flush=True)
for m in mm: m.flush()
