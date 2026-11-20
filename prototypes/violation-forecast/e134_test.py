"""E134, the one pre-registered test read. Nothing here is fitted or selected on test: C and the combination weights are read
from results/e134/frozen.json (e134_dev.py), the classifiers are refitted on train exactly as on dev.

1. Embedding scores on the test eval-v1.2 rows, both configs, every N -> scores/e134-emb-<cfg>-test.jsonl; frozen combination
   a * probe + b * emb -> scores/e134-comb-<cfg>-test.jsonl. Scored afterwards with injecforks_score.py (see E134).
2. Secondary cuts (the card's (a) and (b)) and the dev -> test drop per method, N = 16 / 32, paired case bootstrap.
3. Part B on test: E103's T2 / T4 test rollouts (v1.6, 299 realistic cases x 16) at N = 8 / 16 (the probe's variant scores
   stop at 16); T0 = the dataset's 16 rollouts of the same cases (eval + evalextra; probe e79final + e79final-e113).

  python e134_test.py   -> results/e134/{test-cuts.json, variants-test.json}
"""
import json

import numpy as np

from e134_common import NS, OUT, SC, fit, load, load_variant, read_scores, score_rows
from injecforks_metrics import per_case_auroc

NB = 2000
FZ = json.loads((OUT / "frozen.json").read_text())
C = FZ["C"]


def boot(d, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    xs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(NB)]
    return [round(float(d.mean()), 3)] + [round(float(q), 3) for q in np.percentile(xs, [2.5, 97.5])]


def wc(rows, S, N):
    rs = [r for r in rows if (r["row_id"], N) in S]
    return per_case_auroc([r["case"] for r in rs], [r["violates"] for r in rs], [S[(r["row_id"], N)] for r in rs])


SHARED_TOOLKITS = ("Amazon", "Shopify", "GitHub", "Teladoc")
CUTS = {"all": lambda r: True,
        "(a) no TeladocSendMessage / InventoryManagementSystemSearchSuppliers":
            lambda r: r["user_tool"] not in ("TeladocSendMessage", "InventoryManagementSystemSearchSuppliers"),
        "(b) toolkit absent from train and dev": lambda r: not r["user_tool"].startswith(SHARED_TOOLKITS)}

# 1. test scores
cuts = {}
for cfg in ("smoke", "realistic"):
    rows = load(cfg, "test")
    probe = read_scores(SC / f"e79final-comply-{cfg}-test.jsonl")
    emb = {(r["row_id"], 0): 0.0 for r in rows}
    for N in NS:
        emb.update({(k, N): v for k, v in score_rows(fit(cfg, N, C), f"{cfg}-test", rows, N).items()})
    comb = {k: FZ["comb"][cfg][str(k[1])]["a"] * probe[k] + FZ["comb"][cfg][str(k[1])]["b"] * v for k, v in emb.items()}
    for nm, S in (("emb", emb), ("comb", comb)):
        with open(SC / f"e134-{nm}-{cfg}-test.jsonl", "w") as fh:
            fh.writelines(json.dumps({"row_id": k[0], "N": k[1], "score": v}) + "\n" for k, v in S.items())
    # 2. secondary cuts, emb vs probe vs comb, paired by case
    dev = load(cfg, "dev")
    Sdev = {"emb": read_scores(SC / f"e134-emb-{cfg}-dev.jsonl"), "probe": read_scores(SC / f"e79final-comply-{cfg}-dev.jsonl")}
    for N in (16, 32):
        for cn, keep in CUTS.items():
            rs = [r for r in rows if keep(r)]
            pc = {m: wc(rs, S, N) for m, S in (("emb", emb), ("probe", probe), ("comb", comb))}
            ks = sorted(pc["emb"])
            cuts[f"{cfg} N={N} {cn}"] = dict(cases=len(ks), **{m: round(float(np.mean([pc[m][c] for c in ks])), 3) for m in pc},
                                             emb_minus_probe=boot([pc["emb"][c] - pc["probe"][c] for c in ks]),
                                             comb_minus_probe=boot([pc["comb"][c] - pc["probe"][c] for c in ks]))
        # dev -> test drop per method (independent case samples: CI from separate bootstraps)
        pd_ = {m: wc(dev, Sdev[m], N) for m in Sdev}; pt = {"emb": wc(rows, emb, N), "probe": wc(rows, probe, N)}
        cuts[f"{cfg} N={N} dev->test"] = {m: dict(dev=boot(list(pd_[m].values())), test=boot(list(pt[m].values()))) for m in pd_}
        print(cfg, N, json.dumps({k: v for k, v in cuts.items() if k.startswith(f"{cfg} N={N}")}), flush=True)
(OUT / "test-cuts.json").write_text(json.dumps(cuts, indent=1))

# 3. Part B on test
T0 = load("realistic", "test") + load("realistic", "test", subset="evalextra")
V = {v: load_variant(f"realistic-e103{v}", "test") for v in ("T2", "T4")}
assert len(T0) == 4784 and all(len(V[v]) == 4784 for v in V) and {r["case"] for r in T0} == {r["case"] for r in V["T2"]}
P0 = {**read_scores(SC / "e79final-e113-comply-realistic-test.jsonl"), **read_scores(SC / "e79final-comply-realistic-test.jsonl")}
PV = {v: read_scores(SC / f"e103-9b-comply-realistic-e103{v}-test.jsonl") for v in V}
res = {}
for N in (8, 16):
    clf = fit("realistic", N, C)
    E0 = {(k, N): v for k, v in score_rows(clf, "realistic-test-all16", T0, N).items()}
    EV = {v: {(k, N): s for k, s in score_rows(clf, f"realistic-e103{v}-test", V[v], N).items()} for v in V}
    pc = {("T0", "emb"): wc(T0, E0, N), ("T0", "probe"): wc(T0, P0, N)}
    for v in V:
        pc[(v, "emb")], pc[(v, "probe")] = wc(V[v], EV[v], N), wc(V[v], PV[v], N)
    o = res[N] = {f"{t} {m}": dict(mixed=len(a), auroc=round(float(np.mean(list(a.values()))), 3)) for (t, m), a in pc.items()}
    for v in V:
        both = sorted(set(pc[(v, "emb")]) & set(pc[("T0", "emb")]))
        de = [pc[(v, "emb")][c] - pc[("T0", "emb")][c] for c in both]
        dp = [pc[(v, "probe")][c] - pc[("T0", "probe")][c] for c in both]
        o[f"{v} vs T0"] = dict(cases=len(both), emb_T=round(np.mean([pc[(v, 'emb')][c] for c in both]), 3),
                               emb_T0=round(np.mean([pc[('T0', 'emb')][c] for c in both]), 3),
                               probe_T=round(np.mean([pc[(v, 'probe')][c] for c in both]), 3),
                               probe_T0=round(np.mean([pc[('T0', 'probe')][c] for c in both]), 3),
                               emb_diff=boot(de), probe_diff=boot(dp), emb_minus_probe_diff=boot(np.subtract(de, dp)))
    print(N, json.dumps(o), flush=True)
(OUT / "variants-test.json").write_text(json.dumps(res, indent=1))
