"""E136 gate replay, step 2 (experiments/E136-finetuned-text-baseline.md): E135's analysis (e135_analyze.py, copied; the only
changes add the `ft` gate and its contrasts, and assert E135's emb / comb numbers) with the frozen FT ensemble as a gate. CPU.

  python3 e136_gate_analyze.py   -> results/e136/gate/summary.json

Reads results/e135/{scores.jsonl,thresholds.json} (e135_scores.py, written before any outcome was joined), results/e110/units.jsonl
(outcomes; E110's own gate scores), results/e110/thresholds.json, results/e120/units.jsonl (fresh resample outcomes).
- Reminder arm: E110's `pol` / `within` and bootstrap exactly (2,000 draws of attack cases and look-alike bases, seed 0). Every E110 gate is
  recomputed and asserted against results/e110/summary.json before any E135 number is written.
- Resampling arm: E120's `arms` exactly (case-mean fresh outcome, analytic random gate at the gate's rate, 2,000 draws of attack cases,
  seed 0), asserted against results/e120/summary.json for comply.
- Random reminder gate (analytic, no claim): none - f (none - always) at the gate's attack firing rate f.
"""
import json, subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e136/gate"
B = 2000
E110_GATES = ["comply", "comply_A", "exposure_n0", "exposure_n16", "piguard", "pg2"]
NEW = ["emb", "comb", "ft"]
CUTS = [("calibrated", "0.1"), ("calibrated", "0.2"), ("pooled", "0.1"), ("pooled", "0.2")]


def ci(v):
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def main():
    T110 = json.loads((HERE / "results/e110/thresholds.json").read_text()); T135 = json.loads((OUT / "thresholds.json").read_text())
    S110 = json.loads((HERE / "results/e110/summary.json").read_text()); S120 = json.loads((HERE / "results/e120/summary.json").read_text())
    units = [json.loads(l) for l in open(HERE / "results/e110/units.jsonl")]
    sc = {(r["side"], r["case"], r["rollout"]): r for r in map(json.loads, open(OUT / "scores.jsonl")) if r["set"] == "e110"}
    assert len(sc) == len(units) == 1195
    for u in units:
        s = sc[(u["side"], u["case"], u["rollout"])]; assert close(s["comply"], u["comply"]); u["emb"], u["comb"], u["ft"] = s["emb"], s["comb"], s["ft"]
    A = [x for x in units if x["side"] == "attack"]; L = [x for x in units if x["side"] == "lookalike"]
    ac = np.unique([x["cluster"] for x in A], return_inverse=True)[1]; lc = np.unique([x["cluster"] for x in L], return_inverse=True)[1]
    caseid = np.unique([x["case"] for x in A], return_inverse=True)[1]
    arr = lambda X, k: np.array([np.nan if x.get(k) is None else float(x[k]) for x in X])
    c0, r16 = arr(A, "viol_control"), arr(A, "viol_reminder")
    t0, t16, f0, f16 = arr(L, "task_control"), arr(L, "task_reminder"), arr(L, "follows_control"), arr(L, "follows_reminder")

    def wm(v, w):
        m = ~np.isnan(v); return float(np.sum(v[m] * w[m]) / np.sum(w[m]))

    def pol(g, t, wa, wl):   # E110's pol, token-16 reminder, plus note-following
        fa = arr(A, g) >= t; fl = arr(L, g) >= t; out = np.where(fa, r16, c0)
        none, always = wm(c0, wa), wm(r16, wa); fr = wm(fa.astype(float), wa)
        return dict(call_rate=wm(out, wa), fire_attack=fr, fire_lookalike=wm(fl.astype(float), wl),
                    lookalike_completion=wm(np.where(fl, t16, t0), wl), follows_note=wm(np.where(fl, f16, f0), wl),
                    calls=float(np.sum(wa * out)), avoided=float(np.sum(wa * fa * c0 * (1 - r16))), induced=float(np.sum(wa * fa * (1 - c0) * r16)),
                    random_same_rate=none - fr * (none - always))

    def within(g, t, wa):    # E110's within
        f = (arr(A, g) >= t).astype(float); m = np.bincount(caseid); k = np.bincount(caseid, weights=f); p_ = (k / m)[caseid]
        return wm((1 - f) * c0 + f * r16, wa) - wm((1 - p_) * c0 + p_ * r16, wa), int(((k > 0) & (k < m)).sum())

    rng = np.random.default_rng(0)
    W = [(np.bincount(rng.integers(0, ac.max() + 1, ac.max() + 1), minlength=ac.max() + 1)[ac].astype(float),
          np.bincount(rng.integers(0, lc.max() + 1, lc.max() + 1), minlength=lc.max() + 1)[lc].astype(float)) for _ in range(B)]
    one = (np.ones(len(A)), np.ones(len(L)))
    thr = lambda g, kind, b: (T135 if g in NEW else T110)[kind][b][g]["threshold"]
    KEYS = ("call_rate", "lookalike_completion", "follows_note", "fire_lookalike", "fire_attack")
    res = dict(meta=dict(git=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), attack_runs=len(A),
                         lookalike_runs=len(L), calls_control=int(c0.sum()), bootstrap="E110: 2,000 draws of attack cases and look-alike bases, seed 0"),
               none=dict(call_rate=float(c0.mean()), lookalike_completion=float(np.nanmean(t0)), follows_note=float(np.nanmean(f0))),
               always=dict(call_rate=float(r16.mean()), lookalike_completion=float(np.nanmean(t16)), follows_note=float(np.nanmean(f16))), cut={}, resample={})
    for kind, b in CUTS:
        row = {}
        for g in E110_GATES + NEW:
            t = thr(g, kind, b); bs = [pol(g, t, *w) for w in W]; pa, pl = arr(A, g) >= t, arr(L, g) >= t
            row[g] = dict(threshold=t, **pol(g, t, *one), fire_pooled=float((pa.sum() + pl.sum()) / (len(pa) + len(pl))),
                          precision_attack=float(c0[pa].mean()) if pa.any() else None,
                          avoided_per_100_reminders=float(100 * np.sum(pa * c0 * (1 - r16)) / max(1, pa.sum() + pl.sum())),
                          ci={k: ci([x[k] for x in bs]) for k in KEYS}, _bs=bs)
            if g in E110_GATES:   # reproduce E110's summary before anything new is written
                ref = S110["cut"][f"{kind}|{b}"][g]
                for k in ("call_rate", "fire_attack", "fire_lookalike", "lookalike_completion", "avoided", "induced", "avoided_per_100_reminders"):
                    assert close(row[g][k], ref[k]), (kind, b, g, k, row[g][k], ref[k])
                for k in ("call_rate", "lookalike_completion", "fire_lookalike"):
                    assert all(close(x, y) for x, y in zip(row[g]["ci"][k], ref["ci"][k])), (kind, b, g, k)
        for p_, q in (("emb", "comply"), ("comb", "comply"), ("comb", "emb"), ("emb", "exposure_n0"), ("emb", "piguard"), ("comply", "exposure_n0"),
                       ("ft", "comply"), ("ft", "emb"), ("ft", "exposure_n0"), ("ft", "piguard")):
            for k in ("call_rate", "lookalike_completion", "follows_note"):
                row[f"{p_}-{q}|{k}"] = dict(point=row[p_][k] - row[q][k], ci=ci([x[k] - y[k] for x, y in zip(row[p_]["_bs"], row[q]["_bs"])]))
        for g in ("comply", "emb", "comb", "ft"):
            for k, base in (("lookalike_completion", t0), ("follows_note", f0)):
                row[f"{g}-none|{k}"] = dict(point=row[g][k] - wm(base, one[1]), ci=ci([x[k] - wm(base, w[1]) for x, w in zip(row[g]["_bs"], W)]))
        ref = S110["cut"][f"{kind}|{b}"]["comply-exposure_n0|call_rate"]
        assert close(row["comply-exposure_n0|call_rate"]["point"], ref["point"]) and all(close(x, y) for x, y in zip(row["comply-exposure_n0|call_rate"]["ci"], ref["ci"]))
        for g in ("comply", "emb", "comb", "ft"):
            d, mixed = within(g, thr(g, kind, b), one[0]); row[f"within_{g}"] = dict(diff=d, mixed_cases=mixed, ci=ci([within(g, thr(g, kind, b), w[0])[0] for w in W]))
        wc = S110["cut"][f"{kind}|{b}"]["within_comply"]; assert close(row["within_comply"]["diff"], wc["diff"]) and all(close(x, y) for x, y in zip(row["within_comply"]["ci"], wc["ci"]))
        for g in E110_GATES + NEW:
            row[g].pop("_bs")
        res["cut"][f"{kind}|{b}"] = row
    # resampling, E120's arms
    R = {(u["case"], u["rollout"]): u for u in map(json.loads, open(HERE / "results/e120/units.jsonl"))}
    keep = [u for u in A if (u["case"], u["rollout"]) in R]; assert len(keep) == S120["meta"]["runs"]
    case = np.unique([u["case"] for u in keep], return_inverse=True)[1]; nc = case.max() + 1
    rc0 = np.array([u["viol_control"] for u in keep], float); rs = np.array([R[(u["case"], u["rollout"])]["resample_case_mean"] for u in keep])
    rng = np.random.default_rng(0)
    WR = np.stack([np.ones(len(keep))] + [np.bincount(rng.integers(0, nc, nc), minlength=nc)[case].astype(float) for _ in range(B)])
    names = ["none", "always", "gated", "random"]

    def arms(fire):
        n = WR.sum(1); f = WR @ fire / n; none, always = WR @ rc0 / n, WR @ rs / n; gated = WR @ np.where(fire > 0, rs, rc0) / n
        return np.stack([none, always, gated, none - f * (none - always)], 1), f

    for kind, b in CUTS:
        row = {}
        for g in ("comply", "emb", "comb", "ft", "exposure_n0", "piguard"):
            fire = (np.array([u[g] for u in keep]) >= thr(g, kind, b)).astype(float); G, f = arms(fire); e = G[0]
            d = {nm: dict(point=float(e[i]), ci=ci(G[1:, i])) for i, nm in enumerate(names)}; d["fire_rate"] = float(f[0])
            for a_, b_ in (("gated", "random"), ("gated", "always"), ("gated", "none")):
                i, j = names.index(a_), names.index(b_); d[f"{a_}-{b_}"] = dict(point=float(e[i] - e[j]), ci=ci(G[1:, i] - G[1:, j]))
            d["_G"] = G; row[g] = d
        if f"{kind}|{b}" in S120["cut"]:
            ref = S120["cut"][f"{kind}|{b}"]["case_mean"]
            for k in ("gated", "random", "gated-random", "gated-always"):
                assert close(row["comply"][k]["point"], ref[k]["point"]) and all(close(x, y) for x, y in zip(row["comply"][k]["ci"], ref[k]["ci"])), (kind, b, k)
        gi = names.index("gated")
        for p_ in ("emb", "comb", "ft"):
            row[f"{p_}-comply|gated"] = dict(point=float(row[p_]["_G"][0, gi] - row["comply"]["_G"][0, gi]), ci=ci(row[p_]["_G"][1:, gi] - row["comply"]["_G"][1:, gi]))
        for g in list(row):
            if isinstance(row[g], dict):
                row[g].pop("_G", None)
        res["resample"][f"{kind}|{b}"] = row
    # E135's own numbers for emb / comb reproduce
    S135 = json.loads((HERE / "results/e135/summary.json").read_text())
    for c in res["cut"]:
        for k, v in S135["cut"][c].items():
            if isinstance(v, dict) and "call_rate" in v:
                assert close(res["cut"][c][k]["call_rate"], v["call_rate"]) and all(close(x, y) for x, y in zip(res["cut"][c][k]["ci"]["call_rate"], v["ci"]["call_rate"])), (c, k)
            elif isinstance(v, dict) and "point" in v:
                assert close(res["cut"][c][k]["point"], v["point"]) and all(close(x, y) for x, y in zip(res["cut"][c][k]["ci"], v["ci"])), (c, k)
        for g in ("emb", "comb"):
            assert close(res["resample"][c][g]["gated-random"]["point"], S135["resample"][c][g]["gated-random"]["point"]), (c, g)
    pr = res["cut"]["calibrated|0.1"]["ft-comply|call_rate"]
    verdict = "ft prevents more" if pr["ci"][1] < 0 else "comply prevents more" if pr["ci"][0] > 0 else \
        ("no detectable difference; equivalent at +-0.03" if -0.03 < pr["ci"][0] and pr["ci"][1] < 0.03 else "no detectable difference; not equivalent at +-0.03")
    res["primary"] = dict(contrast="ft - comply, call rate, calibrated 10 %", **pr, verdict=verdict)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["primary"], indent=1))
    for c, row in res["cut"].items():
        print(c)
        for g in ("comply", "emb", "comb", "ft", "exposure_n0", "piguard"):
            r = row[g]
            print(f"  {g:12s} call {r['call_rate']:.3f} [{r['ci']['call_rate'][0]:.3f},{r['ci']['call_rate'][1]:.3f}] fire a/la/all {r['fire_attack']:.2f}/{r['fire_lookalike']:.2f}/{r['fire_pooled']:.2f} "
                  f"prec {r['precision_attack']} avoided/induced {r['avoided']:.0f}/{r['induced']:.0f} per100 {r['avoided_per_100_reminders']:.0f} LA {r['lookalike_completion']:.3f} follows {r['follows_note']:.3f} rand {r['random_same_rate']:.3f}")
        for k in row:
            if "|" in k or k.startswith("within"):
                print("  ", k, json.dumps(row[k]))
    for c, row in res["resample"].items():
        print("resample", c)
        for g in ("comply", "emb", "comb", "ft", "exposure_n0", "piguard"):
            r = row[g]; print(f"  {g:12s} fire {r['fire_rate']:.3f} gated {r['gated']['point']:.3f} random {r['random']['point']:.3f} g-r {r['gated-random']['point']:+.3f} {r['gated-random']['ci']} g-a {r['gated-always']['point']:+.3f} {r['gated-always']['ci']}")
        for p_ in ("emb", "comb", "ft"):
            print("  ", f"{p_}-comply|gated", row[f"{p_}-comply|gated"])


if __name__ == "__main__":
    main()
