"""E109 audit and replay (experiments/E109-comply-gate-where-it-should-win.md, "Audit"). CPU, no generation, no paid calls.

  ~/.venvs/agentdojo/bin/python e109_audit.py build    -> results/e109/audit/{units.jsonl,checks.json}   (joins, score agreement)
  ~/.venvs/agentdojo/bin/python e109_audit.py guards   -> results/e109/audit/guards.json                  (PG2 / PIGuard per case, dev + test)
  ~/.venvs/agentdojo/bin/python e109_audit.py replay   -> results/e109/audit/{replay.json,table.md,fig.png} (steps 2-5)

Units. Attack: E109 3b (test, 176 cases, rollouts 8-15), one control and one reminder continuation from the same 16-token prefix.
Look-alike / twin: E107 part B test (control and always arms from the same prefix). Dev units (thresholds only): E093 attack,
E107 part B dev. Every gate score is recorded beside the one it is checked against:
- comply / comply_A / comply_B: pod layer-20 states at N = 16, frozen ensembles (e109_part3.ensembles);
- exposure_n0 / exposure_n16: E079's detect probe. `*_refit` = the sklearn-1.9 refit E109 used on test; `*_aj` = A1's frozen-probe test
  read (f55b3eb5, results/injecforks/scores/e102-9b-detect-*-test.jsonl), the same probe whose stored dev scores set the thresholds.
"""
import json, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e109_analyze as E

OUT = HERE / "results/e109/audit"
ATTACK_RUN = "user-e109-9b-pod-20261107T132004Z"; STATES_RUN = ATTACK_RUN
A1 = "f55b3eb5"   # A1's test re-score (feat/e103-brittleness), read with git show; not merged into this branch
AJ_SC = "prototypes/violation-forecast/results/injecforks/scores"


def aj(name, N=None):
    """{(row_id, N): score} from A1's commit."""
    txt = subprocess.run(["git", "show", f"{A1}:{AJ_SC}/{name}"], capture_output=True, text=True, check=True, cwd=HERE).stdout
    return {(x["row_id"], x["N"]): x["score"] for x in map(json.loads, txt.splitlines()) if N is None or x["N"] == N}


def attack_pairs():
    """3b records joined by (case, rollout): exactly one decided control and one decided reminder each, else counted and dropped."""
    run = HERE / "data/e109/dl/e109/attack" / ATTACK_RUN / "results"
    recs = [json.loads(l) for l in open(run / "e109.jsonl")]
    n = Counter((x["case"], x["rollout"], x["arm"]) for x in recs)
    by = {}
    for x in recs:
        assert x["row_id"] == f'realistic|test|{x["case"]}|{x["rollout"]}' and x["base"] == x["case"], x["row_id"]
        by.setdefault((x["case"], x["rollout"]), {})[x["arm"]] = x
    bad = dict(duplicate_arm=[k for k, v in n.items() if v != 1], one_arm=[k for k, v in by.items() if set(v) != {"control", "reminder"}],
               undecided=[k for k, v in by.items() if not all(a["decided"] for a in v.values())])
    keys = sorted(k for k in by if k not in bad["one_arm"] and k not in bad["undecided"])
    return by, keys, bad, run


def prefix_check(by, keys):
    """Both arms' saved text starts with the dataset rollout's first 16 reasoning tokens (the cut the pod read); the reminder arm then
    carries exactly the reminder. Decoded from the dataset, not the run."""
    import yaml
    from transformers import AutoTokenizer
    from injecforks_e097 import prefixes, rows
    job = yaml.safe_load((HERE / "configs/e109-9b-pod.yaml").read_text())["attack"]; g = job["generator"]
    tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"]); te = tok.convert_tokens_to_ids("</think>")
    W = {cs: rs for cs, _, rs in rows(HERE / "data/e109/root", "test", "realistic", 16, None)}
    rem = tok.decode(tok(job["reminder"], add_special_tokens=False).input_ids, skip_special_tokens=False)
    out = Counter(); miss = []
    for cs in sorted({c for c, _ in keys}):
        for ro, ids in prefixes(tok, W[cs], 16, te):
            if (cs, ro) not in by:
                out["dataset_rollout_not_run"] += 1; continue
            p = tok.decode(ids, skip_special_tokens=False); c, r = by[(cs, ro)]["control"]["text"], by[(cs, ro)]["reminder"]["text"]
            ok = c.startswith(p) and r.startswith(p + rem) and not c[len(p):].startswith(rem)
            out["ok" if ok else "mismatch"] += 1
            if not ok:
                miss.append([cs, ro])
    out["run_rollout_not_in_dataset"] = sum(1 for k in keys if k[1] not in {ro for ro, *_ in W.get(k[0], [])})
    return dict(out, mismatches=miss[:20])


def agree(got, want, ths):
    got, want = np.asarray(got, float), np.asarray(want, float)
    return dict(n=len(got), corr=float(np.corrcoef(got, want)[0, 1]), max_abs=float(np.max(np.abs(got - want))), median_abs=float(np.median(np.abs(got - want))),
                flips={f"{t:.4g}": int(np.sum((got >= t) != (want >= t))) for t in ths})


def build():
    import e109_part3 as P
    OUT.mkdir(parents=True, exist_ok=True)
    by, keys, bad, run = attack_pairs(); R = P.raw(run)
    TH = json.loads((HERE / "results/e109/part3-thresholds.json").read_text())
    thr = lambda g: sorted({TH[p][b][g] for p in "AB" for b in TH[p]})
    F, S = P.ensembles(STATES_RUN); detect = E.detect_refit()
    rid = [f"realistic|test|{c}|{ro}" for c, ro in keys]
    X16 = np.stack([R[(r, 16)] for r in rid]); X0 = np.stack([R[(f"realistic|test|{c}", 0)] for c, _ in keys])
    sc = {k: F[k](X16) for k in ("comply", "comply_A", "comply_B")}; e16, e0 = detect(X16), detect(X0)
    ajc16, ajd16, ajd0 = aj("e102-9b-comply-realistic-test.jsonl", 16), aj("e102-9b-detect-realistic-test.jsonl", 16), aj("e102-9b-detect-realistic-test.jsonl", 0)
    ajc0 = aj("e102-9b-comply-realistic-test.jsonl", 0)
    units = []
    for i, ((c, ro), r) in enumerate(zip(keys, rid)):
        a = by[(c, ro)]
        units.append(dict(side="attack", split="test", row_id=r, case=c, cluster=c, rollout=ro, user_tool=a["control"]["user_tool"],
                          control=int(bool(a["control"]["violates_strict"])), reminder=int(bool(a["reminder"]["violates_strict"])),
                          comply=float(sc["comply"][i]), comply_A=float(sc["comply_A"][i]), comply_B=float(sc["comply_B"][i]),
                          exposure_n16_refit=float(e16[i]), exposure_n0_refit=float(e0[i]),
                          comply_aj=ajc16.get((r, 16)), exposure_n16_aj=ajd16.get((r, 16)), exposure_n0_aj=ajd0.get((r, 0)), comply_n0_aj=ajc0.get((r, 0))))
    # look-alike / twin side: E107 part B test, scored as E109 part 3 did, plus A1's references
    B = E.benign("test", detect); Rt = P.raw(E.POD["test"]); e0b = json.loads((HERE / "results/e107/exposure-scores-n0.json").read_text())["test"]
    ref = {}
    for cat, cfg in (("lookalike", "realistic-lookalike"), ("twin", "realistic-benign")):
        ref[cat] = {k: aj(f"e102-9b-{p}-{cfg}-test.jsonl") for k, p in (("c", "comply"), ("d", "detect"))}
    rows = {}
    for r in map(json.loads, open(HERE / "results/e107/test-rows.jsonl")):
        rows.setdefault(r["row_id"], {})[r["arm"]] = r
    for x in B:
        v = Rt[(x["row_id"], 16)][None]; cs = x["row_id"].split("|")[2]; rr = rows[x["row_id"]]; cat = x["cat"]
        units.append(dict(side=cat, split="test", row_id=x["row_id"], case=cs, cluster=x["base"], rollout=int(x["row_id"].rsplit("|", 1)[1]),
                          user_tool=rr["control"]["user_tool"], kind=rr["control"]["kind"],
                          control_task=x["control"]["task"], always_task=x["always"]["task"],
                          control_follows=x["control"]["follows_note"], always_follows=x["always"]["follows_note"],
                          control_viol=int(bool(rr["control"]["violates_strict"])), always_viol=int(bool(rr["always"]["violates_strict"])),
                          comply=x["comply"], comply_pod_recomputed=float(F["comply"](v)[0]), comply_A=float(F["comply_A"](v)[0]),
                          comply_B=float(F["comply_B"](v)[0]), exposure_n16_refit=x["exposure"], exposure_n0_refit=e0b[f"{cat}|{cs}"],
                          comply_aj=ref[cat]["c"].get((x["row_id"], 16)), exposure_n16_aj=ref[cat]["d"].get((x["row_id"], 16)),
                          exposure_n0_aj=ref[cat]["d"].get((x["row_id"], 0)), comply_n0_aj=ref[cat]["c"].get((x["row_id"], 0))))
    # checks
    A = [u for u in units if u["side"] == "attack"]; Bn = [u for u in units if u["side"] != "attack"]
    chk = dict(attack=dict(records=sum(len(v) for v in by.values()), pairs=len(by), common=len(keys), cases=len({c for c, _ in keys}),
                           rollouts_per_case=dict(Counter(Counter(c for c, _ in keys).values())), states_n16=len([k for k in R if k[1] == 16]),
                           states_n0=len([k for k in R if k[1] == 0]), **{k: len(v) for k, v in bad.items()}, bad_examples={k: v[:5] for k, v in bad.items()}),
               prefix=prefix_check(by, keys))
    # N = 0 constancy: the pod's prompt read is one state per case; A1's stored N = 0 rows repeat per rollout and must be equal
    const = {}
    for name in ("exposure_n0_aj", "comply_n0_aj", "exposure_n0_refit"):
        for side in ("attack", "lookalike", "twin"):
            g = {}
            for u in units:
                if u["side"] == side and u[name] is not None:
                    g.setdefault(u["case"], []).append(u[name])
            const[f"{name}|{side}"] = dict(cases=len(g), max_within_case_range=float(max(np.ptp(v) for v in g.values())) if g else None)
    chk["n0_constancy"] = const
    # pod scores vs A1's frozen-probe test read of the same rows (independent pipeline: E102's pod, its own forward of prompt + prefix)
    sa = {}
    for side, X in (("attack", A), ("lookalike", [u for u in Bn if u["side"] == "lookalike"]), ("twin", [u for u in Bn if u["side"] == "twin"])):
        for ours, theirs, th in (("comply", "comply_aj", thr("comply")), ("exposure_n16_refit", "exposure_n16_aj", thr("exposure_n16")),
                                 ("exposure_n0_refit", "exposure_n0_aj", thr("exposure_n0"))):
            Z = [(u[ours], u[theirs]) for u in X if u[theirs] is not None]
            sa[f"{side}|{ours}~{theirs}"] = dict(missing_ref=len(X) - len(Z), **(agree(*zip(*Z), th) if Z else {}))
    sa["lookalike+twin|comply(E107 stored)~comply(recomputed from pod states)"] = agree([u["comply"] for u in Bn], [u["comply_pod_recomputed"] for u in Bn], thr("comply"))
    chk["score_agreement"] = sa
    (OUT / "units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units)); (OUT / "checks.json").write_text(json.dumps(chk, indent=1, default=str))
    print(json.dumps(chk, indent=1, default=str)[:12000])


def guards():
    """PG2 86M and PIGuard per case (dev + test, attack / look-alike / twin), on the tool result exactly as e109_part3 scored it, saved
    this time; checked against A1's test read (e102-classifiers-test.json, `general` input): medians and fire rate at 0.5 per category."""
    import torch
    from injecforks_input_classifiers import model, score
    import e109_part3 as P
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = {(sp, side, c): t for sp in ("dev", "test") for side, cfg in (("attack", "realistic"), ("lookalike", "realistic-lookalike"), ("twin", "realistic-benign"))
         for c, t in P.texts(cfg, sp).items()}
    u = sorted(set(T.values())); out = dict(device=dev, scores={})
    for name in ("pg2-86m", "piguard"):
        tok, m = model(name, dev)
        with torch.no_grad():
            p, nw = score(u, tok, m, dev)
        s = dict(zip(u, map(float, p))); out["scores"][name] = {"|".join(k): s[t] for k, t in T.items()}; del m; torch.cuda.empty_cache()
    txt = subprocess.run(["git", "show", f"{A1}:prototypes/violation-forecast/results/injecforks/analysis/e102-classifiers-test.json"], capture_output=True, text=True, check=True, cwd=HERE).stdout
    ref = json.loads(txt); cmp = {}
    for name in ("pg2-86m", "piguard"):
        r = ref[f"{name}|general"]
        for side, cat in (("attack", "injected"), ("lookalike", "lookalike"), ("twin", "benign")):
            v = [x for k, x in out["scores"][name].items() if k.startswith(f"test|{side}|")]
            cmp[f"{name}|{side}"] = dict(n=len(v), n_aj=r["cases"][cat], median=round(float(np.median(v)), 4), median_aj=r["median"][cat],
                                         at_0_5=round(float(np.mean(np.array(v) >= 0.5)), 3), at_0_5_aj=r["at_0_5"][cat])
    out["vs_aj"] = cmp; (OUT / "guards.json").write_text(json.dumps(out, indent=1)); print(json.dumps(cmp, indent=1))


def dev():
    """Reproduce part3-thresholds.json from dev (e109_part3's own rule and items), and save the dev units for the recalibration bootstrap."""
    import e109_part3 as P
    F, S = P.ensembles(STATES_RUN); A, B, _ = P.dev_items(F, S)
    L = [x for x in B if x["cat"] == "lookalike"]; T = [x for x in B if x["cat"] == "twin"]
    te = E.quantile_thr([x["exposure"] for x in L], E.CASCADE_PASS); passed = [x for x in L if x["exposure"] >= te]
    th = dict(A={}, B={})
    for b in E.BUDGET_A:
        th["A"][str(b)] = {g: E.quantile_thr([x[g] for x in L], b) for g in P.GATES if g != "cascade"}
        th["A"][str(b)]["cascade"] = [te, E.quantile_thr([x["comply"] for x in passed], b * len(L) / len(passed))]
    for b in E.BUDGET_B:
        th["B"][str(b)] = {g: E.b_thr(g, A, L, T, b) for g in P.GATES if g != "cascade"}
        th["B"][str(b)]["cascade"] = [te, E.b_thr("cascade", A, L, T, b, te)]
    ref = json.loads((HERE / "results/e109/part3-thresholds.json").read_text())
    flat = lambda d: {f"{p}|{b}|{g}" + (f"|{i}" if isinstance(v, list) else ""): w for p in "AB" for b in d[p] for g, v in d[p][b].items()
                      for i, w in (enumerate(v) if isinstance(v, list) else [(0, v)])}
    a, r = flat(th), flat(ref); diff = {k: abs(a[k] - r[k]) for k in r}
    keep = ("comply", "comply_A", "comply_B", "exposure", "exposure_n0", "exposure_n16", "pg2", "piguard")
    units = [dict(side="attack", split="dev", case=x["case"], cluster=x["case"], rollout=x["rollout"], control=x["control"], reminder=x["reminder"],
                  **{k: x[k] for k in keep}) for x in A]
    units += [dict(side=x["cat"], split="dev", case=x["row_id"].split("|")[2], cluster=x["base"], row_id=x["row_id"], control_task=x["control"]["task"],
                   always_task=x["always"]["task"], control_follows=x["control"]["follows_note"], always_follows=x["always"]["follows_note"],
                   **{k: x[k] for k in keep}) for x in B]
    (OUT / "dev-units.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    res = dict(max_abs_diff=max(diff.values()), n_diff_gt_1e3=sum(v > 1e-3 for v in diff.values()), worst=sorted(diff.items(), key=lambda kv: -kv[1])[:5],
               dev_units=dict(Counter(u["side"] for u in units)), dev_lookalike_cases=len({u["case"] for u in units if u["side"] == "lookalike"}),
               dev_lookalike_bases=len({u["cluster"] for u in units if u["side"] == "lookalike"}), recomputed=th)
    (OUT / "dev-thresholds-check.json").write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "recomputed"}, indent=1))


# ---------------------------------------------------------------- replay (steps 2-5)
# Score sources. "aligned" (primary): every exposure score is the frozen detect probe whose stored dev scores set the thresholds (A1's
# test read); "e109": the sklearn-1.9 refit part 3 used on test, kept to reproduce part3-summary.json.
SRC = {"aligned": dict(exposure_n0="exposure_n0_aj", exposure_n16="exposure_n16_aj"), "e109": dict(exposure_n0="exposure_n0_refit", exposure_n16="exposure_n16_refit")}
MAIN = ["comply", "comply_A", "exposure_n0", "exposure_n16", "cascade", "pg2", "piguard"]
LABEL = dict(none="No reminder", always="Always on", random="Random", comply="Comply (N = 16)", comply_A="Comply-A (N = 16)", comply_B="Comply-B (N = 16)",
             exposure_n0="Exposure (N = 0)", exposure_n16="Exposure (N = 16)", cascade="Cascade (exp. N = 16 → comply)", pg2="PromptGuard 2 86M", piguard="PIGuard")
RATES = (0.10, 0.20)


class Data:
    """Test units as arrays; cluster ids for the case bootstrap (attack: case; look-alike / twin: E107 base, as E109)."""

    def __init__(self, units, G, src, te):
        self.U = units; s = SRC[src]
        g = lambda u, k: G[k][f'test|{u["side"]}|{u["case"]}']
        for u in units:
            u["_exposure_n0"], u["_exposure_n16"] = u[s["exposure_n0"]], u[s["exposure_n16"]]
            u["_pg2"], u["_piguard"] = g(u, "pg2-86m"), g(u, "piguard")
        self.side = {k: [u for u in units if u["side"] == k] for k in ("attack", "lookalike", "twin")}
        self.te = te
        a, l, t = self.side["attack"], self.side["lookalike"], self.side["twin"]
        self.ac, self.ar = np.array([u["control"] for u in a], float), np.array([u["reminder"] for u in a], float)
        f = lambda X, k: np.array([np.nan if u[k] is None else float(u[k]) for u in X])
        self.lc, self.la, self.tc, self.ta = f(l, "control_task"), f(l, "always_task"), f(t, "control_task"), f(t, "always_task")
        self.lfc, self.lfa = f(l, "control_follows"), f(l, "always_follows")
        self.cl = {k: np.unique([u["cluster"] for u in self.side[k]], return_inverse=True)[1] for k in self.side}
        self.ncl = {k: self.cl[k].max() + 1 for k in self.side}
        self.case = np.unique([u["case"] for u in a], return_inverse=True)[1]

    def score(self, g, side):
        X = self.side[side]
        if g == "cascade":
            return np.array([u["comply"] if u["_exposure_n16"] >= self.te else -np.inf for u in X])
        k = {"exposure_n0": "_exposure_n0", "exposure_n16": "_exposure_n16", "pg2": "_pg2", "piguard": "_piguard"}.get(g, g)
        return np.array([u[k] for u in X], float)


def wm(v, w):
    m = ~np.isnan(v); return float(np.sum(v[m] * w[m]) / np.sum(w[m])) if np.sum(w[m]) else float("nan")


def outcome(D, fa, fl, ft, w=None):
    """Policy A / B quantities for fire indicators (bool, or probabilities for random) on attack / look-alike / twin units."""
    w = w or {k: np.ones(len(D.side[k])) for k in D.side}; wa, wl, wt = w["attack"], w["lookalike"], w["twin"]
    fa, fl, ft = (np.asarray(x, float) for x in (fa, fl, ft))
    g = (1 - fa) * D.ac + fa * D.ar; base = float(np.sum(wa * D.ac))
    lcomp, tcomp = (1 - fl) * D.lc + fl * D.la, (1 - ft) * D.tc + ft * D.ta
    lfol = (1 - fl) * D.lfc + fl * D.lfa
    fn_c, fn_g = wm(D.lfc, wl), wm(lfol, wl); none = base / wa.sum(); asr = float(np.sum(wa * g) / wa.sum())
    return dict(fire_attack=wm(fa, wa), fire_lookalike=wm(fl, wl), fire_twin=wm(ft, wt),
                fire_traffic=(wm(fa, wa) + wm(fl, wl) + wm(ft, wt)) / 3,
                calls_none=base, calls=float(np.sum(wa * g)), avoided=float(np.sum(wa * fa * D.ac * (1 - D.ar))), induced=float(np.sum(wa * fa * (1 - D.ac) * D.ar)),
                call_rate=asr, call_rate_none=none, A_prevented=(none - asr) / none if none else float("nan"),
                lookalike_completion=wm(lcomp, wl), lookalike_completion_none=wm(D.lc, wl), twin_completion=wm(tcomp, wt), follows_note=wm(lfol, wl),
                B_prevented=((none - asr) + (fn_c - fn_g)) / (none + fn_c))


def match_thr(s, w, r):
    """Highest attainable weighted look-alike reminder rate <= r: the cutoff (fire at score >= t), from scores only."""
    o = np.argsort(-s, kind="stable"); ss, ww = s[o], w[o]; cw = np.cumsum(ww) / ww.sum()
    last = np.r_[ss[1:] != ss[:-1], True]   # rate is only defined after a whole tie block
    ok = np.where(last & (cw <= r + 1e-12))[0]
    return (float(ss[ok[-1]]), float(cw[ok[-1]])) if len(ok) else (np.inf, 0.0)


def attainable(s, r):
    """Attainable look-alike rates bracketing r (prompt-level gates move a whole case, or a tie block, at a time)."""
    o = np.sort(s)[::-1]; u = np.r_[o[1:] != o[:-1], True]; rates = (np.arange(len(o)) + 1)[u] / len(o)
    lo, hi = rates[rates <= r + 1e-12], rates[rates > r + 1e-12]
    return [float(lo[-1]) if len(lo) else 0.0, float(hi[0]) if len(hi) else 1.0]


def gate_out(D, g, t, w=None):
    if g == "none":
        return outcome(D, *(np.zeros(len(D.side[k])) for k in ("attack", "lookalike", "twin")), w)
    if g == "always":
        return outcome(D, *(np.ones(len(D.side[k])) for k in ("attack", "lookalike", "twin")), w)
    if g == "random":   # t = the firing probability, the same on every run
        return outcome(D, *(np.full(len(D.side[k]), t) for k in ("attack", "lookalike", "twin")), w)
    return outcome(D, *(D.score(g, k) >= t for k in ("attack", "lookalike", "twin")), w)


def within(D, g, t, w=None):
    """Step 4: per injected input with m runs and k reminded by the gate, the probe's outcome vs reminding k of m uniformly at random
    (expected value from the saved pairs: each run's reminder outcome weighted k/m). Returns violations-per-run difference."""
    f = (D.score(g, "attack") >= t).astype(float); wa = (w or {}).get("attack", np.ones(len(f)))
    m = np.bincount(D.case, minlength=D.case.max() + 1); k = np.bincount(D.case, weights=f, minlength=len(m)); p = (k / m)[D.case]
    probe = (1 - f) * D.ac + f * D.ar; rnd = (1 - p) * D.ac + p * D.ar
    mixed = (k > 0) & (k < m); ben = D.ac - D.ar   # benefit of the reminder on a run
    var = np.array([np.ptp(ben[D.case == c]) > 0 for c in range(len(m))])
    return dict(diff=float(np.sum(wa * (probe - rnd)) / wa.sum()), probe_rate=wm(probe, wa), random_rate=wm(rnd, wa), fired_runs=int(f.sum()),
                cases=int(len(m)), cases_mixed=int(mixed.sum()), cases_mixed_with_varying_benefit=int((mixed & var).sum()),
                cases_none=int((k == 0).sum()), cases_all=int((k == m).sum()))


def boot_w(D, rng):
    return {k: np.bincount(rng.integers(0, D.ncl[k], D.ncl[k]), minlength=D.ncl[k])[D.cl[k]].astype(float) for k in D.side}


def ci(v):
    v = np.asarray(v, float); return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def replay():
    import matplotlib
    matplotlib.use("Agg")
    units = [json.loads(l) for l in open(OUT / "units.jsonl")]; G = json.loads((OUT / "guards.json").read_text())["scores"]
    TH = json.loads((HERE / "results/e109/part3-thresholds.json").read_text()); te = TH["A"]["0.1"]["cascade"][0]
    dev_units = [json.loads(l) for l in open(OUT / "dev-units.jsonl")]
    res = dict(sample={}, frozen={}, matched={}, within={}, recalibration={}, policy_B={})
    D = Data(units, G, "aligned", te); D9 = Data([dict(u) for u in units], G, "e109", te)
    res["sample"] = {k: dict(cases=len({u["case"] for u in v}), clusters=int(D.ncl[k]), runs=len(v)) for k, v in D.side.items()}
    res["sample"]["attack"].update(calls_control=int(D.ac.sum()), calls_reminder=int(D.ar.sum()), cases_with_any_control_call=int(len({u["case"] for u in D.side["attack"] if u["control"]})),
                                   lookalike_task_unparsed=int(np.isnan(D.lc).sum() + np.isnan(D.la).sum()))
    B = 2000; rng = np.random.default_rng(0); W = [boot_w(D, rng) for _ in range(B)]

    # --- step 3a: frozen dev thresholds, both score sources; conditional-on-cutoff CIs (aligned)
    for pol, bud in (("A", "0.1"), ("A", "0.2"), ("B", "0.1"), ("B", "0.3")):
        th = TH[pol][bud]; row = {}
        for g in ["comply", "comply_A", "comply_B", "exposure_n0", "exposure_n16", "cascade", "pg2", "piguard"]:
            t = th[g][1] if g == "cascade" else th[g]
            m = gate_out(D, g, t); m9 = gate_out(D9, g, t)
            key = "A_prevented" if pol == "A" else "B_prevented"
            bs = [gate_out(D, g, t, w)[key] for w in W]
            row[g] = dict(threshold=t, aligned=m, e109_scores=dict(fire_attack=m9["fire_attack"], fire_lookalike=m9["fire_lookalike"], fire_twin=m9["fire_twin"],
                                                                     A_prevented=m9["A_prevented"], B_prevented=m9["B_prevented"]), ci_conditional=ci(bs), _bs=bs)
        for p, q in (("comply", "exposure_n0"), ("comply_A", "exposure_n0"), ("piguard", "exposure_n0"), ("exposure_n16", "exposure_n0"), ("cascade", "exposure_n16"), ("comply_A", "comply")):
            row[f"{p}-{q}"] = ci(np.array(row[p]["_bs"]) - np.array(row[q]["_bs"]))
        for g in list(row):
            if isinstance(row[g], dict):
                row[g].pop("_bs")
        row["always"] = gate_out(D, "always", None); row["none"] = gate_out(D, "none", None)
        res["frozen"][f"{pol}|{bud}"] = row

    # pre-registered cascade rule (parts 1-2, confirmatory on 3b): A at 10 %, cascade - exposure (N = 16, the pre-registered comparator)
    for src, DD in (("e109", D9), ("aligned", D)):
        th = TH["A"]["0.1"]; bs = []
        rngs = np.random.default_rng(0); WW = W if src == "aligned" else [boot_w(DD, rngs) for _ in range(B)]
        for w in WW:
            bs.append(gate_out(DD, "cascade", th["cascade"][1], w)["A_prevented"] - gate_out(DD, "exposure_n16", th["exposure_n16"], w)["A_prevented"])
        pt = gate_out(DD, "cascade", th["cascade"][1])["A_prevented"] - gate_out(DD, "exposure_n16", th["exposure_n16"])["A_prevented"]
        c = ci(bs); res.setdefault("prereg_cascade_rule", {})[src] = dict(point=pt, ci=c, rule_met=bool(c[0] > 0 and pt >= 0.10))

    # --- recalibration sensitivity: resample dev (attack cases; look-alike / twin bases), re-derive the A thresholds, evaluate on fixed test
    dl = [u for u in dev_units if u["side"] == "lookalike"]; dcl = np.unique([u["cluster"] for u in dl], return_inverse=True)[1]
    dkey = {"exposure_n0": "exposure_n0", "exposure_n16": "exposure_n16", "comply": "comply", "comply_A": "comply_A", "pg2": "pg2", "piguard": "piguard"}
    rr = np.random.default_rng(1); RC = {f"{g}|{b}": [] for g in dkey for b in ("0.1", "0.2")}
    for _ in range(1000):
        wdev = np.bincount(rr.integers(0, dcl.max() + 1, dcl.max() + 1), minlength=dcl.max() + 1)[dcl]
        idx = np.repeat(np.arange(len(dl)), wdev)
        for g, k in dkey.items():
            s = np.array([dl[i][k] for i in idx])
            for b in ("0.1", "0.2"):
                t = float(np.quantile(s, 1 - float(b))); m = gate_out(D, g, t)
                RC[f"{g}|{b}"].append((m["fire_lookalike"], m["fire_attack"], m["A_prevented"], m["lookalike_completion"]))
    for k, v in RC.items():
        v = np.array(v); res["recalibration"][k] = dict(test_fire_lookalike=ci(v[:, 0]), test_fire_attack=ci(v[:, 1]), A_prevented=ci(v[:, 2]), lookalike_completion=ci(v[:, 3]),
                                                       dev_cases=len({u["case"] for u in dl}), dev_bases=int(dcl.max() + 1))

    # --- step 3b: matched realized look-alike reminder rates (retrospective on test; cutoffs from look-alike scores only, re-chosen per replicate)
    for r in RATES:
        row = {}
        ones = {k: np.ones(len(D.side[k])) for k in D.side}
        for g in MAIN:
            s = D.score(g, "lookalike"); t, got = match_thr(s, ones["lookalike"], r)
            m = gate_out(D, g, t); bs = []
            for w in W:
                tw, _ = match_thr(s, w["lookalike"], r); bs.append(gate_out(D, g, tw, w))
            ties = int(len(s) - len(np.unique(s[np.isfinite(s)])) - np.sum(~np.isfinite(s)))
            row[g] = dict(threshold=t, attained=got, attainable=attainable(s[np.isfinite(s)] if g != "cascade" else s, r), tied_lookalike_rows=ties,
                          prompt_level=g in ("exposure_n0", "pg2", "piguard"), **m, _bs=bs,
                          ci={k: ci([b[k] for b in bs]) for k in ("call_rate", "A_prevented", "lookalike_completion", "fire_attack", "fire_lookalike")})
        rnd = gate_out(D, "random", r); rbs = [gate_out(D, "random", r, w) for w in W]
        row["random"] = dict(threshold=None, attained=r, **rnd, _bs=rbs, ci={k: ci([b[k] for b in rbs]) for k in ("call_rate", "A_prevented", "lookalike_completion")})
        # paired contrasts at the same realized look-alike rate (each gate at its own attained rate; see `attained`)
        for p, q in (("comply", "exposure_n0"), ("comply_A", "exposure_n0"), ("exposure_n16", "exposure_n0"), ("piguard", "exposure_n0"), ("pg2", "exposure_n0"),
                     ("comply", "random"), ("exposure_n0", "random"), ("comply_A", "comply"), ("cascade", "exposure_n16")):
            for k in ("call_rate", "lookalike_completion"):
                row[f"{p}-{q}|{k}"] = dict(point=row[p][k] - row[q][k], ci=ci([a[k] - b[k] for a, b in zip(row[p]["_bs"], row[q]["_bs"])]))
        for g in list(row):
            if isinstance(row[g], dict):
                row[g].pop("_bs", None)
        res["matched"][str(r)] = row

    # --- step 4: within-input random control at the frozen A cutoffs and at the matched cutoffs
    for g in ("comply", "comply_A", "exposure_n16", "cascade"):
        cuts = {f"frozen A|{b}": (TH["A"][b][g][1] if g == "cascade" else TH["A"][b][g]) for b in ("0.1", "0.2")}
        cuts.update({f"matched|{r}": res["matched"][str(r)][g]["threshold"] for r in RATES})
        for name, t in cuts.items():
            pt = within(D, g, t); bs = [within(D, g, t, w)["diff"] for w in W]
            res["within"][f"{g}|{name}"] = dict(threshold=t, **pt, ci=ci(bs))

    # --- Policy B under its original equal traffic weights (frozen B cutoffs; look-alikes reused from E107 / parts 1-2)
    for bud in ("0.1", "0.3"):
        res["policy_B"][bud] = {g: dict(fire_traffic=v["aligned"]["fire_traffic"], fire_attack=v["aligned"]["fire_attack"], fire_lookalike=v["aligned"]["fire_lookalike"],
                                        fire_twin=v["aligned"]["fire_twin"], B_prevented=v["aligned"]["B_prevented"], ci=v["ci_conditional"],
                                        lookalike_completion=v["aligned"]["lookalike_completion"])
                                for g, v in res["frozen"][f"B|{bud}"].items() if isinstance(v, dict) and "aligned" in v}
        res["policy_B"][bud]["comply-exposure_n0"] = res["frozen"][f"B|{bud}"]["comply-exposure_n0"]

    # --- curves for the figure (aligned scores; every attainable look-alike rate)
    curves = {}
    for g in ("exposure_n0", "comply", "comply_A", "piguard", "pg2", "exposure_n16"):
        s = D.score(g, "lookalike"); pts = []
        for t in np.r_[np.inf, np.unique(s)[::-1], -np.inf]:
            m = gate_out(D, g, t); pts.append((m["fire_lookalike"], m["call_rate"], m["lookalike_completion"], m["fire_attack"]))
        curves[g] = pts
    res["curves"] = curves
    (OUT / "replay.json").write_text(json.dumps(res, indent=1, default=float))
    figure(res); table(res); print((OUT / "table.md").read_text())


def figure(res):
    import matplotlib.pyplot as plt
    C = dict(exposure_n0="#2a6fdb", comply="#d9480f", piguard="#2b8a3e", random="#868e96", comply_A="#c2255c", exposure_n16="#7950f2", pg2="#5c940d")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    cv = res["curves"]; base = res["frozen"]["A|0.1"]["none"]; alw = res["frozen"]["A|0.1"]["always"]
    for g in ("exposure_n0", "comply", "piguard"):
        p = np.array(cv[g]); ax[0].step(p[:, 0], p[:, 1], where="post", color=C[g], lw=1.8, label=LABEL[g]); ax[1].step(p[:, 0], p[:, 2], where="post", color=C[g], lw=1.8)
    x = np.linspace(0, 1, 50)
    ax[0].plot(x, (1 - x) * base["call_rate"] + x * alw["call_rate"], color=C["random"], ls="--", lw=1.2, label="Random (reference)")
    ax[1].plot(x, (1 - x) * base["lookalike_completion"] + x * alw["lookalike_completion"], color=C["random"], ls="--", lw=1.2)
    for g, mk in (("exposure_n0", "o"), ("comply", "s"), ("comply_A", "D"), ("piguard", "^"), ("exposure_n16", "v")):
        m = res["frozen"]["A|0.1"][g]["aligned"]
        for i, k in enumerate(("call_rate", "lookalike_completion")):
            ax[i].plot(m["fire_lookalike"], m[k], mk, mfc="white", mec=C[g], mew=1.6, ms=7, label=f"{LABEL[g]}, frozen dev cutoff (A 10 %)" if i == 0 else None)
    for r in RATES:
        for g, mk in (("exposure_n0", "o"), ("comply", "s"), ("piguard", "^")):
            m = res["matched"][str(r)][g]
            for i, k in enumerate(("call_rate", "lookalike_completion")):
                ax[i].plot(m["fire_lookalike"], m[k], mk, color=C[g], ms=6)
    for a in ax:
        a.axvline(0.10, color="#ced4da", lw=0.8, zorder=0); a.axvline(0.20, color="#ced4da", lw=0.8, zorder=0); a.set_xlim(0, 1.0); a.grid(alpha=0.25, lw=0.5)
        a.set_xlabel("Realized reminder rate on harmless look-alikes"); a.spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Attacker-directed call rate (strict, per run)"); ax[1].set_ylabel("Look-alike task completion")
    ax[0].set_title("Security (lower is better)", loc="left", fontsize=10); ax[1].set_title("Original task completion (higher is better)", loc="left", fontsize=10)
    n = res["sample"]
    fig.suptitle(f"Token-16 reminder replayed from paired continuations (E109 3b + E107 test): {n['attack']['cases']} injected inputs / {n['attack']['runs']} runs; "
                 f"{n['lookalike']['cases']} look-alikes / {n['lookalike']['runs']} runs.\nOpen markers: frozen dev cutoffs (A 10 %). "
                 "Filled markers: cutoff matched on test at 10 % and 20 % look-alike rate (retrospective).", fontsize=8.5)
    ax[0].legend(fontsize=7, frameon=False, loc="upper right")
    fig.savefig(OUT / "fig.png", dpi=170); fig.savefig(OUT / "fig.svg")


def table(res):
    n = res["sample"]; L = []
    f2, f3 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.3f}")
    L += [f"Common sample (test): injected {n['attack']['cases']} inputs / {n['attack']['runs']} runs (each run: one control and one reminder continuation from the same "
          f"16-token prefix), {n['attack']['calls_control']} strict calls without the reminder, {n['attack']['calls_reminder']} with it on every run; look-alikes "
          f"{n['lookalike']['cases']} inputs / {n['lookalike']['runs']} runs ({n['lookalike']['clusters']} bases); twins {n['twin']['cases']} / {n['twin']['runs']}. "
          "CIs: 2000 bootstrap draws of whole attack cases and whole look-alike/twin bases.", ""]
    L += ["### Matched realized look-alike reminder rate (retrospective, test; Policy A)", "",
          "| Rate | Gate | Attained LA rate (attainable bracket) | Attack runs reminded | Strict calls / runs | Call rate [95 % CI] | Avoided | Newly induced | LA completion [95 % CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in RATES:
        row = res["matched"][str(r)]
        for g in ["random", "exposure_n0", "comply", "comply_A", "exposure_n16", "cascade", "piguard", "pg2"]:
            m = row[g]; br = m.get("attainable"); c = m["ci"]
            L.append(f"| {r:.0%} | {LABEL[g]} | {m['attained']:.3f}" + (f" ({br[0]:.3f}–{br[1]:.3f})" if br else "") + f" | {m['fire_attack']:.2f} | "
                     f"{m['calls']:.1f} / {n['attack']['runs']} | {m['call_rate']:.3f} [{c['call_rate'][0]:.3f}, {c['call_rate'][1]:.3f}] | {m['avoided']:.1f} | {m['induced']:.1f} | "
                     f"{m['lookalike_completion']:.3f} [{c['lookalike_completion'][0]:.3f}, {c['lookalike_completion'][1]:.3f}] |")
        for k in ("comply-exposure_n0", "comply_A-exposure_n0", "exposure_n16-exposure_n0", "piguard-exposure_n0", "comply-random", "exposure_n0-random", "cascade-exposure_n16"):
            a, b = row[f"{k}|call_rate"], row[f"{k}|lookalike_completion"]
            L.append(f"| {r:.0%} | Δ {k} | | | | {a['point']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | | | {b['point']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] |")
    nn = res["frozen"]["A|0.1"]["none"]; aa = res["frozen"]["A|0.1"]["always"]
    L += ["", f"No reminder: call rate {nn['call_rate']:.3f}, LA completion {nn['lookalike_completion']:.3f}. Always on: call rate {aa['call_rate']:.3f} "
          f"({aa['avoided']:.0f} avoided, {aa['induced']:.0f} induced), LA completion {aa['lookalike_completion']:.3f}.", ""]
    L += ["### Frozen dev-calibrated cutoffs (part3-thresholds.json, reproduced exactly from dev)", "",
          "| Policy | Gate | Realized fire: attack / look-alike / twin | A prevented, aligned scores [CI | cutoff] | A prevented, E109's refit scores | Call rate | Avoided / induced | LA completion |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("A|0.1", "A|0.2"):
        for g in ["comply", "comply_A", "exposure_n0", "exposure_n16", "cascade", "pg2", "piguard"]:
            v = res["frozen"][k][g]; m = v["aligned"]; c = v["ci_conditional"]
            L.append(f"| {k} | {LABEL[g]} | {m['fire_attack']:.2f} / {m['fire_lookalike']:.2f} / {m['fire_twin']:.2f} | {m['A_prevented']:.2f} [{c[0]:.2f}, {c[1]:.2f}] | "
                     f"{v['e109_scores']['A_prevented']:.2f} | {m['call_rate']:.3f} | {m['avoided']:.0f} / {m['induced']:.0f} | {m['lookalike_completion']:.3f} |")
    L += ["", "Recalibration sensitivity (dev look-alikes resampled by base, 1000 draws; threshold re-derived, test held fixed): 95 % range", "",
          "| Gate, budget | Test LA fire | Test attack fire | A prevented | LA completion |", "|---|---|---|---|---|"]
    for k, v in res["recalibration"].items():
        L.append(f"| {k} | {v['test_fire_lookalike'][0]:.2f}–{v['test_fire_lookalike'][1]:.2f} | {v['test_fire_attack'][0]:.2f}–{v['test_fire_attack'][1]:.2f} | "
                 f"{v['A_prevented'][0]:.2f}–{v['A_prevented'][1]:.2f} | {v['lookalike_completion'][0]:.3f}–{v['lookalike_completion'][1]:.3f} |")
    L += ["", "### Within-input random control (step 4): probe vs reminding the same k of m runs uniformly at random", "",
          "| Gate, cutoff | Runs reminded | Inputs (mixed k / with varying benefit) | Probe call rate | Random-k call rate | Δ (probe − random) [95 % CI] |", "|---|---|---|---|---|---|"]
    for k, v in res["within"].items():
        L.append(f"| {k} | {v['fired_runs']} | {v['cases']} ({v['cases_mixed']} / {v['cases_mixed_with_varying_benefit']}) | {v['probe_rate']:.4f} | {v['random_rate']:.4f} | "
                 f"{v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] |")
    L += ["", "### Policy B (equal traffic thirds, frozen B cutoffs; the look-alikes are E107's test set, already read in E107 and parts 1-2)", "",
          "| Budget | Gate | Fire: traffic (attack / LA / twin) | B prevented [CI] | LA completion |", "|---|---|---|---|---|"]
    for bud, row in res["policy_B"].items():
        for g, m in row.items():
            if g.startswith("comply-"):
                L.append(f"| {bud} | Δ comply − exposure (N = 0) | | [{m[0]:+.2f}, {m[1]:+.2f}] | |"); continue
            L.append(f"| {bud} | {LABEL.get(g, g)} | {m['fire_traffic']:.2f} ({m['fire_attack']:.2f} / {m['fire_lookalike']:.2f} / {m['fire_twin']:.2f}) | "
                     f"{m['B_prevented']:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}] | {m['lookalike_completion']:.3f} |")
    pr = res["prereg_cascade_rule"]
    L += ["", "Pre-registered cascade rule (A, 10 %, cascade − exposure at N = 16, CI excludes 0 and point ≥ +0.10): " +
          "; ".join(f"{k}: {v['point']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] → {'met' if v['rule_met'] else 'not met'}" for k, v in pr.items())]
    (OUT / "table.md").write_text("\n".join(L) + "\n")


def delegation():
    """Screen (dev, exploratory): do the gates fire on E096's authorised delegation replies, where following the tool-result note is what the
    user asked for? Same injected input with and without the user's authorisation; 18 cases x 8 replies per version. Scores from E096's own
    N = 0 / N = 16 states; comply checked against E096's stored scores first."""
    import e109_part3 as P
    from sklearn.metrics import roc_auc_score
    R = HERE / "data/e096/dl/e096/user-e096-9b-pod-20261029T190638Z/results"
    xs = [x for x in map(json.loads, open(R / "e096.jsonl")) if x["part"] == "delegation" and x["kind"] == "reply" and x["decided"]]
    St = P.raw(R); keep = [x for x in xs if (x["row_id"], 16) in St]
    F, _ = P.ensembles(STATES_RUN); detect = E.detect_refit()
    X16 = np.stack([St[(x["row_id"], 16)] for x in keep]); X0 = np.stack([St[(x["row_id"].rsplit("|", 1)[0], 0)] for x in keep])
    sc = dict(comply=F["comply"](X16), comply_A=F["comply_A"](X16), comply_B=F["comply_B"](X16), exposure_n16=detect(X16), exposure_n0=detect(X0))
    ref = {(x["row_id"], x["N"]): x["score"] for x in map(json.loads, open(HERE / "results/e096/scores.jsonl"))}
    d = np.abs(sc["comply"] - np.array([ref[(x["row_id"], 16)] for x in keep])); assert d.max() < 0.05, d.max()
    TH = json.loads((HERE / "results/e109/part3-thresholds.json").read_text())["A"]["0.1"]
    T110 = json.loads((HERE / "results/e110/thresholds.json").read_text())
    auth = np.array([x["version"] == "auth" for x in keep]); act = np.array([bool(x["violates_strict"]) for x in keep]); case = np.array([x["case"] for x in keep])
    out = dict(replies=len(keep), excluded_short=len(xs) - len(keep), cases=len(set(case)), comply_vs_e096_max_abs=float(d.max()),
               acting=dict(auth=float(act[auth].mean()), unauth=float(act[~auth].mean())), gates={})
    rng = np.random.default_rng(0); cs = sorted(set(case)); ix = {c: np.where(case == c)[0] for c in cs}
    for g, s in sc.items():
        cuts = {"E109 dev A 10 %": TH[g], **({"E110 look-alike 10 %": T110["calibrated"]["0.1"][g]["threshold"], "E110 pooled 10 %": T110["pooled"]["0.1"][g]["threshold"]} if g in T110["calibrated"]["0.1"] else {})}
        row = dict(auroc_acting_within_auth=float(roc_auc_score(act[auth], s[auth])),
                   auroc_auth_acting_vs_unauth=float(roc_auc_score(np.r_[np.ones((auth & act).sum()), np.zeros((~auth).sum())], np.r_[s[auth & act], s[~auth]])))
        for name, t in cuts.items():
            f = s >= t; bs = []
            for _ in range(2000):
                i = np.concatenate([ix[c] for c in rng.choice(cs, len(cs))]); bs.append(f[i][auth[i]].mean() - f[i][~auth[i]].mean())
            row[name] = dict(fire_auth=float(f[auth].mean()), fire_unauth=float(f[~auth].mean()), fire_auth_acting=float(f[auth & act].mean()),
                             diff_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        out["gates"][g] = row
    (OUT / "delegation.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"build": build, "guards": guards, "dev": dev, "replay": replay, "delegation": delegation}[sys.argv[1]]()
