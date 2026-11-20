import json, sys, numpy as np, torch
from pathlib import Path
from collections import defaultdict
HERE = Path(__file__).resolve().parent
VF = HERE.parent
ST = VF / "data/e144-states"
RES = VF / "results/e144"
sys.path.insert(0, str(VF))
from injecforks_metrics import per_case_auroc
IDX = [json.loads(l) for l in open(ST / "index.jsonl")]
CFG = np.array([m["cfg"] for m in IDX]); SP = np.array([m["split"] for m in IDX]); NN = np.array([m["N"] for m in IDX])
Y = np.array([m["y"] for m in IDX]); CASE = np.array([m["case"] for m in IDX]); RID = np.array([m["row_id"] for m in IDX])
NS = [8, 16, 32, 64, 128]; SEL = [8, 16, 32, 64]
_L = {}
def layer(l):
    if l not in _L:
        _L[l] = np.load(ST / f"L{l}.npy")
    return _L[l]
def mask(cfg, sp, ns=NS):
    m = (SP == sp) & np.isin(NN, ns)
    return m & (CFG == cfg) if cfg != "both" else m
def wc(idx, s, n):
    i = idx[NN[idx] == n]; pc = per_case_auroc(CASE[i], Y[i], s[np.isin(idx, i)] if False else s_at(idx, s, i))
    return (float(np.mean(list(pc.values()))) if pc else float("nan")), len(pc)
def s_at(idx, s, i):
    pos = {k: j for j, k in enumerate(idx)}; return s[[pos[k] for k in i]]
def per_n(idx, s):
    """within-case mean AUROC per N and mixed-case count."""
    out = {}
    for n in NS:
        sel = NN[idx] == n
        pc = per_case_auroc(CASE[idx][sel], Y[idx][sel], s[sel])
        out[n] = (float(np.mean(list(pc.values()))) if pc else float("nan"), len(pc))
    return out
def sel_score(res):
    """res: {cfg: per_n dict}; case-count-weighted pool over arms, mean over SEL N (E79's selection rule)."""
    v = []
    for n in SEL:
        xs = [res[c][n] for c in res if res[c][n][1]]
        v.append(sum(m * k for m, k in xs) / sum(k for _, k in xs))
    return float(np.mean(v))
