"""E114 analysis (experiments/E114-risk-trained-comply-probe.md). CPU, no calls, no generation.

  ~/.venvs/agentdojo/bin/python e114_analyze.py   -> results/e114/summary.json

comply_R = comply_A's rows, recipe and N sampling (e109_part3.ensembles) with each attack row's target the share of its input's 8 train
rollouts that comply; look-alikes 0. comply_R0 = one row per input on the prompt state (N = 0), same targets. Soft targets: each row
twice (y = 1 weight p, y = 0 weight 1 - p), classes reweighted to equal total weight. Cutoffs and policy: E110's rules on E110's units
(results/e110/units.jsonl), comply_A and exposure_n0 at their stored E110 cutoffs. Primary: pooled 10 %, call rate comply_R - comply_A,
95 % CI from 2,000 draws of whole attack cases / look-alike bases, seed 0.
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e109_analyze as E
import e109_audit as AU
import e109_part3 as P

OUT = HERE / "results/e114"; E110 = HERE / "results/e110"; RUN110 = "user-e110-9b-pod-20261109T121244Z"
RATE, B = 0.10, 2000
GATES = ["comply_A", "exposure_n0", "comply_R", "comply_R0"]
CONTRASTS = [("comply_R", "comply_A"), ("comply_R0", "comply_A"), ("comply_R0", "exposure_n0")]


def soft_fit(X, p, C):
    """Logistic regression on soft targets p in [0, 1], balanced by total weight per class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); Z = sc.transform(X); p = np.asarray(p, float)
    w = np.r_[p / p.sum(), (1 - p) / (1 - p).sum()]
    c = LogisticRegression(C=C, max_iter=3000).fit(np.r_[Z, Z], np.r_[np.ones(len(p)), np.zeros(len(p))], sample_weight=w * len(w) / 2)
    return sc, c


def probes():
    from e096_analyze import C, LAYER_IDX, SEEDS
    from injecforks_load import load
    from injecforks_probe import train_pairs
    F, S = P.ensembles(AU.STATES_RUN)
    inj = load("realistic", "train", subset=None); la = P.train_lookalikes()
    by = defaultdict(list)
    for r in inj:
        by[r["row_id"].rsplit("|", 1)[0]].append(int(r["violates"]))
    assert all(len(v) == 8 for v in by.values())
    risk = {k: float(np.mean(v)) for k, v in by.items()}
    p = [risk[r["row_id"].rsplit("|", 1)[0]] for r in inj] + [0.0] * len(la)
    fs = []
    for k in range(SEEDS):
        X = S.get(train_pairs(inj, S, k, "realistic|comply") + train_pairs(la, S, k, "e109|lookalike"), LAYER_IDX); fs.append(soft_fit(X, p, C))
    ens = lambda fs: lambda X: np.mean([c.decision_function(s.transform(np.asarray(X, np.float32))) for s, c in fs], 0)
    F["comply_R"] = ens(fs)
    lc = sorted({r["row_id"].rsplit("|", 1)[0] for r in la}); ic = sorted(risk)
    X0 = S.get([(c + "|0", 0) for c in ic + lc], LAYER_IDX)
    r0 = ens([soft_fit(X0, [risk[c] for c in ic] + [0.0] * len(lc), C)])
    F["comply_R0"] = lambda X: np.round(r0(X), 5)   # one prompt state per input: batched float noise (< 1e-6) must not split an input
    train = dict(attack_inputs=len(ic), lookalike_inputs=len(lc), attack_rows=len(inj), lookalike_rows=len(la),
                 input_risk_zero=float(np.mean([risk[c] == 0 for c in ic])), input_risk_mean=float(np.mean(list(risk.values()))))
    print("train", train)
    return F, train


def main():
    from sklearn.metrics import roc_auc_score
    from e107_n0 import e102_states
    from injecforks_probe import States
    F, train = probes()
    # calibration sets (scores only): E107 test look-alike runs, E109 3b attack runs
    AUu = [json.loads(l) for l in open(AU.OUT / "units.jsonl")]
    La = [u for u in AUu if u["side"] == "lookalike"]; Aa = [u for u in AUu if u["side"] == "attack"]
    Rt = P.raw(E.POD["test"]); R3 = P.raw(HERE / "data/e109/dl/e109/attack" / AU.ATTACK_RUN / "results"); S102 = States([e102_states()])
    case = lambda rid: rid.rsplit("|", 1)[0]
    cal = dict(comply_R=(F["comply_R"](np.stack([Rt[(u["row_id"], 16)] for u in La])), F["comply_R"](np.stack([R3[(u["row_id"], 16)] for u in Aa]))),
               comply_R0=(F["comply_R0"](S102.get([(u["row_id"], 0) for u in La], 3)), F["comply_R0"](np.stack([R3[(case(u["row_id"]), 0)] for u in Aa]))))
    TH = json.loads((E110 / "thresholds.json").read_text())
    th = {k: {g: TH[k]["0.1"][g] for g in ("comply_A", "exposure_n0")} for k in ("calibrated", "pooled")}
    for g, (sl, sa) in cal.items():
        th["calibrated"][g] = dict(zip(("threshold", "calibration_rate"), AU.match_thr(sl, np.ones(len(sl)), RATE)))
        th["pooled"][g] = dict(zip(("threshold", "calibration_rate"), AU.match_thr(np.r_[sa, sl], np.ones(len(sa) + len(sl)), RATE)))
    # E110 fresh units, new scores
    U = [json.loads(l) for l in open(E110 / "units.jsonl")]; St = P.raw(HERE / "data/e110/dl/e110" / RUN110 / "results")
    rid = lambda u: f'{"realistic" if u["side"] == "attack" else "realistic-lookalike-e110"}|test|{u["case"]}|{u["rollout"]}'
    X16 = np.stack([St[(rid(u), 16)] for u in U]); X0 = np.stack([St[(case(rid(u)), 0)] for u in U])
    chk = F["comply_A"](X16); ref = np.array([u["comply_A"] for u in U]); corr = float(np.corrcoef(chk, ref)[0, 1])
    print("comply_A rebuild vs E110 stored: corr", corr); assert corr > 0.999
    for u, a, b in zip(U, F["comply_R"](X16), F["comply_R0"](X0)):
        u["comply_R"], u["comply_R0"] = float(a), float(b)
    A = [u for u in U if u["side"] == "attack"]; L = [u for u in U if u["side"] == "lookalike"]
    arr = lambda X, k: np.array([np.nan if x.get(k) is None else float(x[k]) for x in X])
    c0, r16 = arr(A, "viol_control"), arr(A, "viol_reminder"); t0, t16 = arr(L, "task_control"), arr(L, "task_reminder")
    ac = np.unique([x["cluster"] for x in A], return_inverse=True)[1]; lc = np.unique([x["cluster"] for x in L], return_inverse=True)[1]
    caseid = np.unique([x["case"] for x in A], return_inverse=True)[1]
    wm = lambda v, w: float(np.sum((v * w)[~np.isnan(v)]) / np.sum(w[~np.isnan(v)]))
    def pol(g, t, wa, wl):
        fa = arr(A, g) >= t; fl = arr(L, g) >= t
        return dict(call_rate=wm(np.where(fa, r16, c0), wa), fire_attack=wm(fa.astype(float), wa), fire_lookalike=wm(fl.astype(float), wl),
                    lookalike_completion=wm(np.where(fl, t16, t0), wl), fire_pooled=float((fa.sum() + fl.sum()) / (len(fa) + len(fl))),
                    avoided_per_100_reminders=float(100 * np.sum(fa * c0 * (1 - r16)) / max(1, fa.sum() + fl.sum())))
    def within(g, t):
        f = (arr(A, g) >= t).astype(float); m = np.bincount(caseid); k = np.bincount(caseid, weights=f); p = (k / m)[caseid]
        return float(np.mean((1 - f) * c0 + f * r16) - np.mean((1 - p) * c0 + p * r16))
    rng = np.random.default_rng(0)
    W = [(np.bincount(rng.integers(0, ac.max() + 1, ac.max() + 1), minlength=ac.max() + 1)[ac].astype(float),
          np.bincount(rng.integers(0, lc.max() + 1, lc.max() + 1), minlength=lc.max() + 1)[lc].astype(float)) for _ in range(B)]
    one = (np.ones(len(A)), np.ones(len(L))); ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    res = dict(train=train, rebuild_corr=corr, thresholds=th, none=float(c0.mean()), always=float(r16.mean()), cut={}, contrasts={})
    for kind in ("calibrated", "pooled"):
        res["cut"][kind] = {g: dict(threshold=th[kind][g]["threshold"], **pol(g, th[kind][g]["threshold"], *one), within_vs_random_k=within(g, th[kind][g]["threshold"]))
                            for g in GATES}
        bs = {g: np.array([pol(g, th[kind][g]["threshold"], *w)["call_rate"] for w in W]) for g in GATES}
        res["contrasts"][kind] = {f"{a} - {b}": dict(point=res["cut"][kind][a]["call_rate"] - res["cut"][kind][b]["call_rate"], ci=ci(bs[a] - bs[b])) for a, b in CONTRASTS}
    pr = res["contrasts"]["pooled"]["comply_R - comply_A"]
    pr["verdict"] = "comply_R lower" if pr["ci"][1] < 0 else "comply_R higher" if pr["ci"][0] > 0 else "no detectable difference"
    res["primary"] = dict(contrast="pooled 10 %, call rate comply_R - comply_A", **pr)
    # exploratory (not pre-registered): a prompt-read gate can place the reminder in the prompt (E110's reminder_n0 arm)
    rn0, tn0 = arr(A, "viol_reminder_n0"), arr(L, "task_reminder_n0")
    res["exploratory_prompt_reminder"] = {}
    for kind in ("calibrated", "pooled"):
        t = th[kind]["comply_R0"]["threshold"]; fa = arr(A, "comply_R0") >= t; fl = arr(L, "comply_R0") >= t
        bs = np.array([wm(np.where(fa, rn0, c0), w[0]) - pol("comply_A", th[kind]["comply_A"]["threshold"], *w)["call_rate"] for w in W])
        cr = float(np.mean(np.where(fa, rn0, c0)))
        res["exploratory_prompt_reminder"][kind] = dict(call_rate=cr, lookalike_completion=wm(np.where(fl, tn0, t0), one[1]),
                                                        minus_comply_A=cr - res["cut"][kind]["comply_A"]["call_rate"], ci=ci(bs))
    print("exploratory comply_R0 + prompt reminder", res["exploratory_prompt_reminder"])
    # accuracy: run label among attack runs; harmful call vs look-alike; input level (case mean score vs case comply share)
    y = c0.astype(int); harm = y == 1
    cm = lambda v: np.bincount(caseid, weights=v) / np.bincount(caseid)
    res["auroc"] = {g: dict(among_attack_runs=float(roc_auc_score(y, arr(A, g))),
                            harmful_vs_lookalike=float(roc_auc_score(np.r_[np.ones(harm.sum()), np.zeros(len(L))], np.r_[arr(A, g)[harm], arr(L, g)])),
                            input_any_call=float(roc_auc_score(cm(c0) > 0, cm(arr(A, g)))))
                    for g in GATES}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    (OUT / "scores.jsonl").write_text("".join(json.dumps({k: u[k] for k in ("side", "case", "rollout", "comply_R", "comply_R0")}) + "\n" for u in U))
    print(json.dumps({k: res[k] for k in ("primary", "contrasts", "auroc")}, indent=1))
    for kind in res["cut"]:
        for g, x in res["cut"][kind].items():
            print(kind, g, {k: round(v, 3) for k, v in x.items()})


if __name__ == "__main__":
    main()
