"""E136 step 2, dev only (before the test read): freeze the setting, train the 3-seed ensemble, dev scores, combination, look-alikes.

1. Frozen setting = the (lr, class weight, epoch) with the highest mean dev within-case AUROC over every grid cell (e136_grid.py).
2. Seeds 0, 1, 2 at that setting for each config and N in argv (default 16 32 8 64 128), saved fp16 (e136_common.model_path).
   Final score = the mean of the three seeds' logit differences (score_ens), the same function the test read uses.
3. Dev: scores/e136-ft-<cfg>-dev.jsonl (N = 0 constant 0, the empty prefix, as E134); per-seed and ensemble within-case AUROC.
4. Probe + FT combination on dev exactly as E134 (conditional logit, case fixed effects, within-case-demeaned scores / dev SD),
   frozen as a * probe + b * ft per (config, N). Dev comb scores are in-sample.
5. Look-alike separation on dev (exploratory): pooled AUROC of the violating dev eval rows vs every <cfg>-lookalike dev row, FT
   vs the comply probe (E102's e102-9b-comply scores for both sides, so the probe reads both from one run) vs E134's e5.

  python e136_final.py [N ...]  -> results/e136/{frozen.json, dev-final.json}, scores/e136-{ft,comb}-<cfg>-dev.jsonl
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.discrete.conditional_models import ConditionalLogit

from e136_common import EMAX, MODEL, OUT, SC, gpu_lock, load, model_path, read_scores, score_ens, train
from injecforks_metrics import within_case_mean

SEEDS = [0, 1, 2]
NS_RUN = [int(a) for a in sys.argv[1:]] or [16, 32, 8, 64, 128]
CFGS = ("smoke", "realistic")

# 1. freeze
G = [json.loads(l) for l in open(OUT / "dev-grid.jsonl")]
cells = {(g["cfg"], g["N"]) for g in G}
assert len(G) == 6 * len(cells), "grid incomplete"
setting = {}
for g in G:
    for ep, a in g["auroc"].items():
        setting.setdefault((g["lr"], g["cw"], int(ep)), []).append(a)
assert all(len(v) == len(cells) for v in setting.values())
mean = {k: float(np.mean(v)) for k, v in setting.items()}
LR, CW, EP = max(mean, key=mean.get)
FZ = OUT / "frozen.json"
fz = json.loads(FZ.read_text()) if FZ.exists() else {}
fz.update(model=MODEL, text="reasoning[:cut[N]]", lr=LR, cw=CW, epochs=EP, schedule_epochs=EMAX, seeds=SEEDS, probe="e79final-comply",
          grid_cells=sorted(map(list, cells)), mean_by_setting={f"lr={k[0]} cw={k[1]} ep={k[2]}": v for k, v in sorted(mean.items(), key=lambda x: -x[1])})
fz.setdefault("comb", {})
print("frozen", LR, CW, EP, round(mean[(LR, CW, EP)], 4), flush=True)

# 2-5
dev = {c: load(c, "dev") for c in CFGS}
la = {c: load(f"{c}-lookalike", "dev", subset=None) for c in CFGS}
rep = json.loads((OUT / "dev-final.json").read_text()) if (OUT / "dev-final.json").exists() else {}
for N in NS_RUN:
    for cfg in CFGS:
        for s in SEEDS:
            if not model_path(cfg, N, s).exists():
                with gpu_lock():
                    train(cfg, N, LR, CW, s, stop=EP, save=model_path(cfg, N, s))
        with gpu_lock():
            E = score_ens(cfg, N, SEEDS, dev[cfg] + la[cfg])
        b = [r for r in dev[cfg] if r["cut"].get(N) is not None]
        cs, ys = [r["case"] for r in b], [r["violates"] for r in b]
        probe = read_scores(SC / f"e79final-comply-{cfg}-dev.jsonl")
        emb = read_scores(SC / f"e134-emb-{cfg}-dev.jsonl")
        o = dict(rows=len(b), ft=within_case_mean(cs, ys, [E[r["row_id"]][0] for r in b]),
                 ft_seeds=[within_case_mean(cs, ys, [E[r["row_id"]][1][i] for r in b]) for i in range(len(SEEDS))],
                 probe=within_case_mean(cs, ys, [probe[(r["row_id"], N)] for r in b]), emb=within_case_mean(cs, ys, [emb[(r["row_id"], N)] for r in b]))
        d = pd.DataFrame(dict(case=cs, y=[int(v) for v in ys], probe=[probe[(r["row_id"], N)] for r in b], ft=[E[r["row_id"]][0] for r in b]))
        dm = d.groupby("case")[["probe", "ft"]].transform(lambda x: x - x.mean()); sd = dm.std()
        f = ConditionalLogit(d.y, dm / sd, groups=d.case).fit(disp=0)
        a_, b_ = float(f.params["probe"] / sd["probe"]), float(f.params["ft"] / sd["ft"])
        fz["comb"].setdefault(cfg, {"0": dict(a=1.0, b=0.0)})[str(N)] = dict(a=a_, b=b_, p_probe=float(f.pvalues["probe"]), p_ft=float(f.pvalues["ft"]))
        o["comb_in_sample"] = within_case_mean(cs, ys, list(a_ * d.probe + b_ * d.ft))
        # look-alikes
        P102 = {**read_scores(SC / f"e102-9b-comply-{cfg}-dev.jsonl"), **read_scores(SC / f"e102-9b-comply-{cfg}-lookalike-dev.jsonl")}
        vi = [r for r in b if r["violates"]]; lr_ = [r for r in la[cfg] if r["cut"].get(N) is not None and (r["row_id"], N) in P102]
        yl = [1] * len(vi) + [0] * len(lr_)
        o["lookalike"] = dict(violating=len(vi), lookalike_rows=len(lr_), lookalike_rows_total=sum(r["cut"].get(N) is not None for r in la[cfg]),
                              ft=roc_auc_score(yl, [E[r["row_id"]][0] for r in vi + lr_]),
                              probe=roc_auc_score(yl, [P102[(r["row_id"], N)] for r in vi + lr_]))
        rep[f"{cfg} N={N}"] = o
        print(cfg, N, json.dumps(o), flush=True)
        fn = SC / f"e136-ft-{cfg}-dev.jsonl"
        old = [x for x in map(json.loads, open(fn))] if fn.exists() else []
        old = [x for x in old if x["N"] not in (0, N)]
        new = [{"row_id": r["row_id"], "N": 0, "score": 0.0} for r in dev[cfg]] + [{"row_id": r["row_id"], "N": N, "score": E[r["row_id"]][0]} for r in b]
        allr = sorted(old + new, key=lambda x: (x["N"], x["row_id"]))
        fn.write_text("".join(json.dumps(x) + "\n" for x in allr))
        cf = SC / f"e136-comb-{cfg}-dev.jsonl"
        oldc = [x for x in map(json.loads, open(cf))] if cf.exists() else []
        oldc = [x for x in oldc if x["N"] not in (0, N)]
        newc = [{"row_id": r["row_id"], "N": 0, "score": probe[(r["row_id"], 0)]} for r in dev[cfg]] + \
               [{"row_id": r["row_id"], "N": N, "score": a_ * probe[(r["row_id"], N)] + b_ * E[r["row_id"]][0]} for r in b]
        cf.write_text("".join(json.dumps(x) + "\n" for x in sorted(oldc + newc, key=lambda x: (x["N"], x["row_id"]))))
        FZ.write_text(json.dumps(fz, indent=1)); (OUT / "dev-final.json").write_text(json.dumps(rep, indent=1))
