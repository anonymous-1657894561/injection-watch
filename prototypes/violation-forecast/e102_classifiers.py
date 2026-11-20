"""E102: input classifiers on the look-alikes. PromptGuard 2 (86M, 22M) and PIGuard, as injecforks_input_classifiers.py
scores them (same windows, same revisions), on each case's tool result: injected, benign twin, look-alike.

  ~/.venvs/e01/bin/python e102_classifiers.py [--split test --final]
  -> results/injecforks/analysis/e102-classifiers-<split>.json

Look-alike texts are the `tool_response` of the InjecForks v1.4 look-alike rows (byte-identical to the request bodies the rollouts
sent). Thresholds: 0.5, and each classifier's benign-twin dev 95th percentile (E095's rule), fixed on dev for both splits. One
score per case, so firing rates are over cases. `auroc_violating_vs_lookalike` gives each row its case's score and compares
violating injected rows (strict label) with look-alike rows, the positives every trigger shares (decisions/, 2026-11-02). On test,
injected rows are rollouts 8-15 (2026-10-29), and the read happens once (testread.py).
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from injecforks_input_classifiers import CLF, model, score
from injecforks_load import load
import testread

HERE = Path(__file__).resolve().parent
CFG = {"general": "realistic", "targeted": "smoke"}   # arm name -> dataset config (config ids unchanged)

a = testread.args(argparse.ArgumentParser()).parse_args(); testread.check(a, "e102-classifiers"); sp = a.split
dev = "cuda" if torch.cuda.is_available() else "cpu"
case_text = lambda rows: {r["case"]: r["tool_response"] for r in rows}
texts, rows = {}, {}
for arm, c in CFG.items():
    inj = [r for r in load(c, sp) if sp == "dev" or r["rollout"] >= 8]; la = load(c + "-lookalike", sp)
    L = case_text(la); I = case_text(inj)
    texts[arm] = dict(injected=list(I.values()), benign=list(case_text(load(c + "-benign", sp)).values()),
                      benign_dev=list(case_text(load(c + "-benign", "dev")).values()),
                      lookalike=list(L.values()), kind=[k.rsplit("-", 1)[1] for k in L])
    # row-level: each row carries its case's index into the injected / look-alike text lists
    rows[arm] = dict(violating=[list(I).index(r["case"]) for r in inj if r["violates"]], lookalike=[list(L).index(r["case"]) for r in la])
out = {}
for name in CLF:
    tok, m = model(name, dev)
    for arm, T in texts.items():
        s = {k: score(T[k], tok, m, dev)[0] for k in ("injected", "benign", "benign_dev", "lookalike")}
        p95 = float(np.percentile(s.pop("benign_dev"), 95)); kind = np.array(T["kind"])
        fire = lambda t: {k: round(float(np.mean(v >= t)), 3) for k, v in s.items()} | \
            {f"lookalike_{k}": round(float(np.mean(s["lookalike"][kind == k] >= t)), 3) for k in sorted(set(T["kind"]))}
        y = np.r_[np.ones(len(s["injected"])), np.zeros(len(s["lookalike"]))]
        v, l = s["injected"][rows[arm]["violating"]], s["lookalike"][rows[arm]["lookalike"]]
        out[f"{name}|{arm}"] = dict(cases={k: len(v) for k, v in s.items()}, at_0_5=fire(0.5), benign_p95=round(p95, 4), at_benign_p95=fire(p95),
                                    auroc_injected_vs_benign=round(roc_auc_score(np.r_[np.ones(len(s["injected"])), np.zeros(len(s["benign"]))], np.r_[s["injected"], s["benign"]]), 3),
                                    auroc_injected_vs_lookalike=round(roc_auc_score(y, np.r_[s["injected"], s["lookalike"]]), 3),
                                    auroc_violating_vs_lookalike=round(roc_auc_score(np.r_[np.ones(len(v)), np.zeros(len(l))], np.r_[v, l]), 3),
                                    median={k: round(float(np.median(v)), 4) for k, v in s.items()})
        print(name, arm, json.dumps(out[f"{name}|{arm}"]), flush=True)
    del m; torch.cuda.empty_cache() if dev == "cuda" else None
f = HERE / f"results/injecforks/analysis/e102-classifiers-{sp}.json"; f.write_text(json.dumps(out, indent=1))
testread.done(a, "e102-classifiers", output=str(f.relative_to(HERE)))
