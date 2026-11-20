"""E124: reporting audit of E115 (experiments/E124-e115-audit.md). CPU only, reads E115's saved units; no new generation.

  ~/.venvs/agentdojo/bin/python e124_audit.py   -> results/e124/{summary.json,table.md,per_carrier.md,units.jsonl,fig_e124.png}

Two columns per policy: deployment (E115's frozen pooled thresholds, fire 0/1) and exact (E115's 10 % weighted replay, fire_exact,
recomputed in every draw). E115's fire_exact / metrics / random_k are imported unchanged. Prevented / induced are paired within a run:
control arm vs the reminded arm of the same run, as expected counts under the policy's fire probability.
"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e115_analyze as A

OUT = HERE / "results/e124"; W, RATE = A.W, A.RATE; B = int(__import__("os").environ.get("E124_B", 2000))
# policy -> (score, reminder arm); comply_R0_n16 holds the prompt score fixed and moves the reminder to token 16
GATES = dict(comply_R0=("comply_R0", "reminder_n0"), exposure_n0=("exposure_n0", "reminder_n0"), exposure_M=("exposure_M", "reminder_n0"),
             risk_M=("risk_M", "reminder_n0"), comply_A=("comply_A", "reminder"), r0a=("r0a", "reminder"), monitor=("monitor", "reminder"),
             comply_R0_n16=("comply_R0", "reminder"))
CONTRASTS = [("primary", "comply_R0", "comply_A"),            # E115's primary (each gate with its own reminder)
             ("target", "risk_M", "exposure_M"),               # (i) training target, same states/search, prompt reminder
             ("timing", "comply_R0", "comply_R0_n16"),         # (ii) same prompt score, prompt vs token-16 reminder
             ("early_reasoning", "r0a", "comply_R0_n16"),      # (iii) prompt score with vs without the token-16 state, token-16 reminder
             ("vs_exposure", "comply_R0", "exposure_n0"), ("monitor", "monitor", "comply_R0_n16"), ("within", "comply_A", "random_k_A")]
KEYS = ("call_rate", "completion")
BIG = "GoogleMapGetLocationDetails"


def load():
    U = [json.loads(l) for l in open(HERE / "results/e115/units.jsonl")]
    TH = json.loads((HERE / "results/e115/frozen/thresholds.json").read_text())["gates"]
    return U, TH


def policies(U, sc, w, th, mult):
    """{name: (fire prob, reminder arm)}; th None = exact 10 % replay, else frozen thresholds"""
    m = np.ones(len(U)) if mult is None else mult; P = {}
    for g, (s, rem) in GATES.items():
        x = sc[s]
        P[g] = (A.fire_exact(x, w * m, RATE) if th is None else (np.nan_to_num(x, nan=-np.inf) >= th[s]).astype(float), rem)
    P["random_k_A"] = (A.random_k(U, P["comply_A"][0]), "reminder")
    # uniform random reminding at the pooled rate the column uses (10 % exact; comply_R0's realised rate at deployment is not matched)
    P["uniform_n0"] = (np.full(len(U), RATE), "reminder_n0"); P["uniform_n16"] = (np.full(len(U), RATE), "reminder")
    P["never"] = (np.zeros(len(U)), "reminder"); P["always_n0"] = (np.ones(len(U)), "reminder_n0"); P["always_n16"] = (np.ones(len(U)), "reminder")
    return P


def counts(U, p, rem, mult=None):
    """raw paired counts (expected under fractional firing): prevented / induced harmful calls on attack runs, harmful calls on benign runs"""
    m = np.ones(len(U)) if mult is None else mult
    side = np.array([u["side"] for u in U]); a = side == "attack"
    c0 = np.array([u["viol_control"] for u in U], float); r = np.array([u[f"viol_{rem}"] for u in U], float)
    return dict(prevented=float(np.sum((m * p * c0 * (1 - r))[a])), induced=float(np.sum((m * p * (1 - c0) * r)[a])),
                untreated=float(np.sum((m * c0)[a])), treated_policy=float(np.sum((m * (p * r + (1 - p) * c0))[a])),
                benign_calls_policy=float(np.sum((m * (p * r + (1 - p) * c0))[~a])),
                reminders={s: float(np.sum((m * p)[side == s])) for s in W}, runs={s: float(np.sum(m[side == s])) for s in W})


def column(U, sc, w, th=None, mult=None):
    P = policies(U, sc, w, th, mult)
    return {k: dict(**A.metrics(U, p, rem, w, mult), counts=counts(U, p, rem, mult), reminder=rem) for k, (p, rem) in P.items()}, P


def main():
    U, TH = load(); TH = {g: TH[g]["pooled"]["threshold"] for g in TH}
    side = np.array([u["side"] for u in U]); n = {s: int((side == s).sum()) for s in W}
    w = np.array([W[u["side"]] / n[u["side"]] for u in U])
    sc = {g: np.array([u[g] for u in U], float) for g in ("comply_R0", "exposure_n0", "exposure_M", "risk_M", "comply_A", "r0a", "monitor")}
    src = np.unique([u["source_task"] for u in U], return_inverse=True)[1]
    cars, car = np.unique([u["carrier"] for u in U], return_inverse=True)
    cols = dict(deployment=TH, exact=None)
    res = dict(meta=dict(units=len(U), runs=n, inputs={s: len({u["case"] for u in U if u["side"] == s}) for s in W}, source_tasks=int(src.max() + 1),
                         carriers=list(map(str, cars)), weights=W, rate=RATE, thresholds=TH,
                         note="reanalysis of E115's already-read units; deployment contrasts are not budget-matched"))
    point, Pfire = {}, {}
    for c, th in cols.items():
        point[c], Pfire[c] = column(U, sc, w, th)
    res["policies"] = point

    def diffs(r):
        return {f"{nm}|{k}": r[a][k] - r[b][k] for nm, a, b in CONTRASTS for k in KEYS}

    def boot(groups, mult0=None, seed=0):
        rng = np.random.default_rng(seed); K = groups.max() + 1; D = {c: [] for c in cols}; G = {c: [] for c in cols}
        for _ in range(B):
            mult = np.bincount(rng.integers(0, K, K), minlength=K)[groups].astype(float)
            if mult0 is not None:
                mult = mult * mult0
            for c, th in cols.items():
                r, _ = column(U, sc, w, th, mult); D[c].append(diffs(r)); G[c].append({g: {k: r[g][k] for k in KEYS} for g in r})
        return D, G

    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    Ds, Gs = boot(src); Dc, _ = boot(car)
    res["gate_ci_source"] = {c: {g: {k: ci([x[g][k] for x in Gs[c]]) for k in KEYS} for g in point[c]} for c in cols}
    res["contrasts"] = {c: {key: dict(point=v, ci_source=ci([d[key] for d in Ds[c]]), ci_carrier=ci([d[key] for d in Dc[c]]))
                            for key, v in diffs(point[c]).items()} for c in cols}
    # leave one carrier out: points for every contrast, and source-task CIs without the big carrier
    loco = {}
    for i, cname in enumerate(cars):
        mult = (car != i).astype(float)
        loco[str(cname)] = {c: diffs(column(U, sc, w, th, mult)[0]) for c, th in cols.items()}
    res["leave_one_carrier_out"] = loco
    keep = (cars[car] != BIG).astype(float)
    Dx, _ = boot(src, mult0=keep)
    res["without_" + BIG] = {c: {key: dict(point=loco[BIG][c][key], ci_source=ci([d[key] for d in Dx[c]])) for key in loco[BIG][c]} for c in cols}
    # general = source CI excludes 0 and sign holds in every LOCO fit
    for c in cols:
        for key, v in res["contrasts"][c].items():
            s = np.sign(v["point"]); excl = v["ci_source"][0] > 0 or v["ci_source"][1] < 0
            v["sign_holds_every_loco"] = bool(all(np.sign(loco[k][c][key]) == s for k in loco))
            v["general"] = bool(excl and v["sign_holds_every_loco"])
    # check against E115's published exact numbers
    E = json.loads((HERE / "results/e115/summary.json").read_text())
    res["check_vs_e115"] = {g: dict(e115=E["exact"][g]["call_rate"], e124=point["exact"][g]["call_rate"],
                                    match=abs(E["exact"][g]["call_rate"] - point["exact"][g]["call_rate"]) < 5e-4)
                            for g in ("comply_R0", "exposure_n0", "exposure_M", "risk_M", "comply_A", "r0a", "monitor")}
    res["check_vs_e115"]["comply_R0_n16"] = dict(e115=E["exact"]["comply_R0_n16"]["call_rate"], e124=point["exact"]["comply_R0_n16"]["call_rate"],
                                                 match=abs(E["exact"]["comply_R0_n16"]["call_rate"] - point["exact"]["comply_R0_n16"]["call_rate"]) < 5e-4)
    # per carrier
    pc = {}
    for i, cname in enumerate(cars):
        mk = car == i; a = mk & (side == "attack")
        row = dict(attack_inputs=len({U[j]["case"] for j in np.where(a)[0]}), attack_runs=int(a.sum()),
                   benign_runs=int((mk & (side != "attack")).sum()), untreated_harmful=int(sum(U[j]["viol_control"] for j in np.where(a)[0])))
        for c in cols:
            for g in ("comply_R0", "comply_A", "exposure_M", "risk_M", "comply_R0_n16", "r0a"):
                p, rem = Pfire[c][g]; m = mk.astype(float); k = counts(U, p, rem, m)
                row[f"{c}|{g}"] = dict(prevented=k["prevented"], induced=k["induced"], reminders_attack=k["reminders"]["attack"])
        pc[str(cname)] = row
    res["per_carrier"] = pc
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    with open(OUT / "units.jsonl", "w") as f:
        for j, u in enumerate(U):
            f.write(json.dumps(dict(row_id=u["row_id"], side=u["side"], carrier=u["carrier"], source_task=u["source_task"],
                                    **{f"{c}|{k}": round(float(Pfire[c][k][0][j]), 6) for c in cols for k in GATES})) + "\n")
    tables(res); figure(res)


def tables(res):
    f = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    P, CI = res["policies"], res["gate_ci_source"]; R = res["meta"]["runs"]
    order = ["never", "comply_R0", "exposure_n0", "exposure_M", "risk_M", "comply_A", "random_k_A", "comply_R0_n16", "r0a", "monitor",
             "uniform_n0", "uniform_n16", "always_n0", "always_n16"]
    L = [f"Runs: attack {R['attack']}, twin {R['twin']}, look-alike {R['lookalike']}. Counts are paired (control vs reminded arm of the same run), "
         "expected under fractional firing. Harmful = strict call. CIs: 2,000 draws of whole source tasks.", ""]
    for c, title in (("deployment", "Deployment: E115's frozen pooled thresholds (not budget-matched)"), ("exact", "Exact 10 % weighted replay")):
        L += [f"### {title}", "", "| Policy | reminder | reminded attack / twin / look-alike (rate) | pooled | harmful / attack run [CI] | prevented | induced | "
              "completion weighted [CI] | completion att / twin / look | net avoided / 100 rem. |", "|---|---|---|---|---|---|---|---|---|---|"]
        for g in order:
            m = P[c][g]; k = m["counts"]; ci = CI[c].get(g, {})
            cc = lambda key: f" [{ci[key][0]:.3f}, {ci[key][1]:.3f}]" if key in ci else ""
            L.append(f"| {g} | {'prompt' if m['reminder'] == 'reminder_n0' else 'token 16'} | "
                     + " / ".join(f"{k['reminders'][s]:.0f} ({m['fire'][s]:.3f})" for s in W) + f" | {m['fire_pooled']:.3f} | "
                     f"{k['treated_policy']:.1f}/{k['runs']['attack']:.0f} = {f(m['call_rate'])}{cc('call_rate')} | {k['prevented']:.1f} | {k['induced']:.1f} | "
                     f"{f(m['completion'])}{cc('completion')} | " + " / ".join(f(m["completion_by_class"][s]) for s in W) + f" | {f(m['net_avoided_per_100_reminders'])} |")
        L += ["", "| Contrast | harmful calls [source CI] [carrier CI] | completion [source CI] | general? |", "|---|---|---|---|"]
        for nm, a, b in CONTRASTS:
            v, q = res["contrasts"][c][f"{nm}|call_rate"], res["contrasts"][c][f"{nm}|completion"]
            L.append(f"| {nm} ({a} − {b}) | {v['point']:+.4f} [{v['ci_source'][0]:+.4f}, {v['ci_source'][1]:+.4f}] [{v['ci_carrier'][0]:+.4f}, {v['ci_carrier'][1]:+.4f}] | "
                     f"{q['point']:+.4f} [{q['ci_source'][0]:+.4f}, {q['ci_source'][1]:+.4f}] | {'yes' if v['general'] else 'no'}"
                     + ("" if v["sign_holds_every_loco"] else " (sign flips in a LOCO fit)") + " |")
        L.append("")
    L += ["Check vs E115 (exact harmful-call rate): " + ", ".join(f"{g} {v['e124']:.4f}{'' if v['match'] else ' MISMATCH ' + format(v['e115'], '.4f')}"
                                                            for g, v in res["check_vs_e115"].items())]
    (OUT / "table.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    # per carrier + LOCO
    cars = res["meta"]["carriers"]; pc = res["per_carrier"]
    M = ["| Carrier | attack inputs | attack runs | benign runs | untreated harmful | " + " | ".join(f"{c} {g} prev / ind" for c in ("deployment", "exact")
                                                                                      for g in ("comply_R0", "comply_A", "exposure_M", "risk_M", "comply_R0_n16", "r0a")) + " |",
         "|" + "---|" * 17]
    for cn in cars:
        r = pc[cn]
        M.append(f"| {cn} | {r['attack_inputs']} | {r['attack_runs']} | {r['benign_runs']} | {r['untreated_harmful']} | "
                 + " | ".join(f"{r[c + '|' + g]['prevented']:.1f} / {r[c + '|' + g]['induced']:.1f}" for c in ("deployment", "exact")
                              for g in ("comply_R0", "comply_A", "exposure_M", "risk_M", "comply_R0_n16", "r0a")) + " |")
    M += ["", "Leave one carrier out (harmful calls per attack run; contrast point):", "",
          "| Carrier left out | column | " + " | ".join(nm for nm, _, _ in CONTRASTS) + " |", "|" + "---|" * (2 + len(CONTRASTS))]
    for cn in cars:
        for c in ("deployment", "exact"):
            M.append(f"| {cn} | {c} | " + " | ".join(f"{res['leave_one_carrier_out'][cn][c][nm + '|call_rate']:+.4f}" for nm, _, _ in CONTRASTS) + " |")
    X = res["without_" + BIG]
    M += ["", f"Without {BIG}, source-task CIs:", "", "| Contrast | deployment | exact |", "|---|---|---|"]
    for nm, _, _ in CONTRASTS:
        M.append(f"| {nm} | " + " | ".join(f"{X[c][nm + '|call_rate']['point']:+.4f} [{X[c][nm + '|call_rate']['ci_source'][0]:+.4f}, {X[c][nm + '|call_rate']['ci_source'][1]:+.4f}]"
                                           for c in ("deployment", "exact")) + " |")
    (OUT / "per_carrier.md").write_text("\n".join(M) + "\n"); print("\n".join(M))


def figure(res):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    COL = dict(comply_R0="#7048e8", comply_A="#c2255c", exposure_n0="#2a6fdb", exposure_M="#1098ad", risk_M="#5f3dc4", r0a="#e8590c",
               monitor="#2b8a3e", comply_R0_n16="#b197fc")
    P = res["policies"]; fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for g, col in COL.items():
        for c, mk in (("deployment", "x"), ("exact", "o")):
            m = P[c][g]
            ax[0].plot(m["fire_pooled"], m["call_rate"], mk, color=col, ms=7, mec="white" if mk == "o" else None, label=g if c == "exact" else None)
            ax[1].plot(m["fire_pooled"], m["completion"], mk, color=col, ms=7, mec="white" if mk == "o" else None)
        ax[0].plot([P["deployment"][g]["fire_pooled"], P["exact"][g]["fire_pooled"]], [P["deployment"][g]["call_rate"], P["exact"][g]["call_rate"]], color=col, lw=0.8, alpha=0.6)
    for a in ax:
        a.axvline(RATE, color="#adb5bd", ls=":", lw=1); a.grid(alpha=0.25, lw=0.5); a.set_xlabel("Realised weighted pooled reminder rate")
    ax[0].axhline(P["exact"]["never"]["call_rate"], color="#868e96", ls="--", lw=1)
    ax[1].axhline(P["exact"]["never"]["completion"], color="#868e96", ls="--", lw=1)
    ax[0].set_ylabel("Harmful calls per attack run"); ax[1].set_ylabel("Task completion (traffic-weighted)")
    ax[0].set_title("E115 audit: x = frozen deployment threshold, o = exact 10 % replay", loc="left", fontsize=9)
    ax[1].set_title("Completion at the same operating points (dashed = no reminder)", loc="left", fontsize=9)
    ax[0].legend(fontsize=7, frameon=False)
    fig.savefig(OUT / "fig_e124.png", dpi=170); fig.savefig(OUT / "fig_e124.svg")


if __name__ == "__main__":
    main()
