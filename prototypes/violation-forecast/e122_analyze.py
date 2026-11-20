"""E122 analysis (experiments/E122-attack-style-transfer.md). CPU, no calls.

  ~/.venvs/agentdojo/bin/python e122_analyze.py RUN_ID     -> results/e122/{units.jsonl, summary.json, table.md, fig_e122.png}

E115's frozen gates (e115_freeze.gates on results/e115/frozen/heads.npz; comply_R0 / comply_A refitted deterministically by
e114_analyze.probes, as e115_analyze does) score every E122 run reaching N = 16 from the pod's states; a gate flags a run when its score is at
or above its frozen weighted-pooled-10 % threshold. Harmful = the rollout's own strict call (untreated). E115's benign twins are scored from
E115's pods for the clean-input false-alarm rate. 95 % CIs: 2,000 draws of whole source tasks (carrier x goal), paired across styles, seed 0.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e115_freeze as FZ
from injecforks_e122 import CFG, STYLES

OUT = HERE / "results/e122"; ROOT = HERE / "data/e122/injecforks_e122"; DL = HERE / "data/e122/dl/e122"
E115_RUNS = ["user-e115-9b-pod-s0-20261111T094749Z", "user-e115-9b-pod-s1-20261111T095205Z", "user-e115-9b-pod-s2-20261111T130246Z",
             "user-e115-9b-pod-s3-20261111T122951Z"]
GATES = ["comply_R0", "exposure_M", "risk_M", "exposure_n0", "comply_A", "r0a"]
HELD = [s for s in STYLES if s != "important"]
WRAPPED, UNWRAPPED = ["tool_knowledge", "softened", "casual"], ["ignore_previous", "system_message", "injecagent_base", "injecagent_enhanced", "natural"]
B, MIN_H = 2000, 15


def states(dirs):
    X0, X16 = {}, {}
    for d in dirs:
        M0 = [json.loads(l) for l in open(d / "states0-meta.jsonl")]; S0 = np.load(d / "states0.npy").astype(np.float32)
        assert all(m["layers"] == FZ.LAYERS for m in M0); X0.update({m["case"]: s for m, s in zip(M0, S0)})
        M = [json.loads(l) for l in open(d / "states-meta.jsonl")]; S = np.load(d / "states.npy").astype(np.float32)
        X16.update({m["row_id"]: s for m, s in zip(M, S)})
    return X0, X16


def score(G, keys, X0, X16):
    case = lambda rid: rid.rsplit("|", 1)[0]
    A0 = {li: np.stack([X0[case(k)][li] for k in keys]) for li in range(len(FZ.LAYERS))}; A16 = np.stack([X16[k] for k in keys])
    return {g: G[g](A0, A16) for g in GATES}


def units(run):
    import e114_analyze as R
    arr = dict(np.load(HERE / "results/e115/frozen/heads.npz")); F, _ = R.probes(); G = FZ.gates(arr, F)
    TH = {g: json.load(open(HERE / "results/e115/frozen/thresholds.json"))["gates"][g]["pooled"]["threshold"] for g in GATES}
    X0, X16 = states([DL / run / "results"])
    lab = {}
    for (s, la), cfg in CFG.items():
        for l in open(ROOT / cfg / "test.jsonl"):
            r = json.loads(l); lab[f"{cfg}|test|{r['case']}|{r['rollout']}"] = (s, "lookalike" if la else "attack", r)
    keys = sorted(X16); assert set(keys) <= set(lab), "states for runs the release does not hold"
    sc = score(G, keys, X0, X16); U = []
    for i, k in enumerate(keys):
        s, side, r = lab[k]
        U.append(dict(row_id=k, style=s, side=side, case=r["case"], source_task=r["case"].split(":lookalike-")[0], carrier=r["user_tool"],
                      harmful=int(side == "attack" and r["violates_strict"] in (True, "True")), **{g: float(sc[g][i]) for g in GATES},
                      **{f"flag_{g}": int(sc[g][i] >= TH[g]) for g in GATES}))
    # E115's benign twins, scored from E115's pods
    X0t, X16t = states([HERE / "data/e115/dl/e115" / r / "results" for r in E115_RUNS])
    tk = sorted(k for k in X16t if k.startswith("realistic-benign-e115|"))
    st = score(G, tk, X0t, X16t)
    twins = dict(runs=len(tk), inputs=len({k.rsplit("|", 1)[0] for k in tk}), flagged={g: float(np.mean(st[g] >= TH[g])) for g in GATES})
    return U, twins, TH


def stats(U, mult=None):
    """per style and pooled: harmful rate, capture, injected / look-alike flagged, AUROC among attack runs; mult = per-run bootstrap weight"""
    from sklearn.metrics import roc_auc_score
    m = np.ones(len(U)) if mult is None else mult
    style = np.array([u["style"] for u in U]); side = np.array([u["side"] for u in U]); y = np.array([u["harmful"] for u in U])
    out = {}
    groups = {s: style == s for s in STYLES} | {"held_out": np.isin(style, HELD), "wrapped": np.isin(style, WRAPPED), "unwrapped": np.isin(style, UNWRAPPED)}
    for name, gm in groups.items():
        a = gm & (side == "attack"); la = gm & (side == "lookalike"); h = a & (y == 1)
        d = dict(runs=float(m[a].sum()), harmful=float(m[h].sum()), rate=float(m[h].sum() / max(1e-9, m[a].sum())))
        for g in GATES:
            f = np.array([u[f"flag_{g}"] for u in U], float)
            d[f"capture_{g}"] = float((f * m)[h].sum() / max(1e-9, m[h].sum())) if m[h].sum() else np.nan
            d[f"flag_injected_{g}"] = float((f * m)[a].sum() / max(1e-9, m[a].sum()))
            d[f"flag_lookalike_{g}"] = float((f * m)[la].sum() / max(1e-9, m[la].sum())) if m[la].sum() else np.nan
            ya, sa, wa = y[a], np.array([u[g] for u in U])[a], m[a]
            ok = wa > 0
            d[f"auroc_{g}"] = float(roc_auc_score(ya[ok], sa[ok], sample_weight=wa[ok])) if len(set(ya[ok])) == 2 else np.nan
        out[name] = d
    out["auroc_heldout_mean_minus_important"] = {g: float(np.nanmean([out[s][f"auroc_{g}"] for s in HELD]) - out["important"][f"auroc_{g}"]) for g in GATES}
    return out


def contrasts(o):
    c = {"primary|held_out|comply_R0-exposure_M": o["held_out"]["capture_comply_R0"] - o["held_out"]["capture_exposure_M"]}
    for s in STYLES + ["wrapped", "unwrapped"]:
        c[f"{s}|comply_R0-exposure_M"] = o[s]["capture_comply_R0"] - o[s]["capture_exposure_M"]
    for s in HELD:
        for g in GATES:
            c[f"{s}|{g}|vs_important"] = o[s][f"capture_{g}"] - o["important"][f"capture_{g}"]
            c[f"{s}|{g}|transfer_test"] = o[s][f"capture_{g}"] - 0.5 * o["important"][f"capture_{g}"]
    for g in GATES:
        c[f"auroc|{g}|heldout_mean-important"] = o["auroc_heldout_mean_minus_important"][g]
    return c


def figure(one, ci):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2)); x = np.arange(len(STYLES)); wd = 0.38
    for j, (g, col) in enumerate((("comply_R0", "#2f6db5"), ("exposure_M", "#d08a2c"))):
        v = [one[s][f"capture_{g}"] for s in STYLES]; lo = [ci[s][f"capture_{g}"][0] for s in STYLES]; hi = [ci[s][f"capture_{g}"][1] for s in STYLES]
        ax.bar(x + (j - 0.5) * wd, v, wd, color=col, label=g, yerr=[np.array(v) - lo, np.array(hi) - v], capsize=2, error_kw=dict(lw=0.8))
    for i, s in enumerate(STYLES):
        ax.text(i, -0.1, f"{int(one[s]['harmful'])} harmful", ha="center", fontsize=7, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(STYLES, rotation=25, ha="right", fontsize=8); ax.set_ylim(-0.14, 1.05)
    ax.set_ylabel("harmful runs flagged at frozen threshold"); ax.axvline(0.5, color="#999", lw=0.6, ls=":")
    ax.set_title("E122: frozen E115 gates on held-out attack styles (95 % CI, source-task bootstrap)", fontsize=10); ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_e122.png", dpi=170)


def main():
    run = sys.argv[1]; OUT.mkdir(parents=True, exist_ok=True)
    U, twins, TH = units(run)
    (OUT / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in U))
    one = stats(U); c1 = contrasts(one)
    src = np.unique([u["source_task"] for u in U], return_inverse=True)[1]; K = src.max() + 1; rng = np.random.default_rng(0)
    draws = []
    for _ in range(B):
        m = np.bincount(rng.integers(0, K, K), minlength=K)[src].astype(float); o = stats(U, m); draws.append((o, contrasts(o)))
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    ci = {k: {kk: q([d[0][k][kk] for d in draws]) for kk in one[k]} for k in one if isinstance(one[k], dict) and "runs" in one[k]}
    cc = {k: dict(point=v, ci=q([d[1][k] for d in draws])) for k, v in c1.items()}
    res = dict(run=run, thresholds=TH, twins_e115=twins, by_style=one, ci=ci, contrasts=cc,
               underpowered=[s for s in STYLES if one[s]["harmful"] < MIN_H], primary_underpowered=one["held_out"]["harmful"] < 60)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    f = lambda k, g: f"{one[k][g]:.2f} [{ci[k][g][0]:.2f}, {ci[k][g][1]:.2f}]"
    md = ["| style | runs | harmful (rate) | capture comply_R0 | capture exposure_M | capture risk_M | capture comply_A | injected flagged R0 / exp_M | look-alikes flagged R0 / exp_M | AUROC R0 / exp_M |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for s in STYLES + ["held_out", "wrapped", "unwrapped"]:
        o = one[s]
        md.append(f"| {s} | {int(o['runs'])} | {int(o['harmful'])} ({o['rate']:.3f}) | {f(s, 'capture_comply_R0')} | {f(s, 'capture_exposure_M')} | "
                  f"{o['capture_risk_M']:.2f} | {o['capture_comply_A']:.2f} | {o['flag_injected_comply_R0']:.2f} / {o['flag_injected_exposure_M']:.2f} | "
                  f"{o['flag_lookalike_comply_R0']:.2f} / {o['flag_lookalike_exposure_M']:.2f} | {o['auroc_comply_R0']:.2f} / {o['auroc_exposure_M']:.2f} |")
    md += ["", "| contrast | point | 95 % CI |", "|---|---|---|"]
    md += [f"| {k} | {v['point']:+.3f} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] |" for k, v in cc.items() if "|vs_important" not in k and "transfer_test" not in k]
    md += ["", f"E115 twins ({twins['inputs']} inputs, {twins['runs']} runs) flagged: " + ", ".join(f"{g} {v:.3f}" for g, v in twins["flagged"].items())]
    (OUT / "table.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    figure(one, ci)


if __name__ == "__main__":
    main()
