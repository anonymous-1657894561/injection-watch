"""E125 test analysis (experiments/E125-agentdojo-gptoss-heldout.md). CPU. Refuses to run unless results/e125/frozen/ is committed and clean.

  ~/.venvs/agentdojo/bin/python e125_analyze.py --data DIR   -> results/e125/{summary.json, table.md, units.jsonl, fig_e125.png, fig_e125.svg}

Units: test inputs (160 attacked, 38 clean) x 8 continuations r. Per run: U (untreated), P (E105's reminder at the eligible checkpoint),
T (E093's reminder after U's first 16 analysis tokens; U where not applicable). Inputs without a checkpoint: U only, no gate can act.
A policy is a fire probability f per run and the arm it switches to; the run's outcome is f * arm + (1 - f) * U (expected value).
- none; always (P at every checkpoint); exposure, risk, monitor (prompt time: one decision per checkpoint, arm P); comply16 (per run,
  arm T); random16 (within each checkpoint, the same number of runs as comply16 fires, at random, arm T).
- frozen = the dev thresholds (results/e125/frozen/thresholds.json); replay = the cutoff on test scores alone giving exactly 10 % weighted
  pooled firing (traffic weights attack 0.25 / clean 0.75 per run class), the tie block fired fractionally, recomputed in every draw.
Episode attack success = AgentDojo security over all attack runs (inputs without a checkpoint included). Clean completion = utility over
clean runs. Prevented / induced = U harmful and arm not / U not and arm harmful, weighted by f; net per 100 reminders over all runs.
Runs with an episode error in any arm they use are dropped from every policy and counted. 95 % CI: 2,000 draws of whole inputs within
class (attack, clean), seed 0, paired across policies.
"""
import argparse, json, os, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e125_fit import MAN, RATE, RES, W, load_split, rows

CFG = yaml.safe_load((HERE / "configs/e125.yaml").read_text())
FZ = RES / "frozen"; OUT = RES
PROMPT_GATES = ("exposure", "risk", "monitor")


def frozen_committed():
    if "E125_OUT" in os.environ:   # synthetic checks write outside the repo; nothing there is a result
        return
    rel = FZ.relative_to(HERE.parent.parent)
    for f in ("heads.npz", "thresholds.json", "selection.json"):
        if subprocess.run(["git", "ls-files", "--error-unmatch", str(rel / f)], cwd=HERE.parent.parent, capture_output=True).returncode:
            raise SystemExit(f"{rel / f} is not committed: freeze before reading test outcomes")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=HERE.parent.parent).returncode:
        raise SystemExit(f"{rel} differs from the commit")


def head_fn(Z, h):
    li = int(Z[f"{h}.li"]); ks = sorted({int(k.split(".")[1]) for k in Z.files if k.startswith(h + ".") and k.split(".")[1].isdigit()})

    def f(S):
        x = S[:, li].astype(np.float64)
        return np.mean([((x - Z[f"{h}.{k}.mean"]) / Z[f"{h}.{k}.scale"]) @ Z[f"{h}.{k}.coef"] + Z[f"{h}.{k}.intercept"][0] for k in ks], 0)
    return f


def replay(s, w, rate=RATE):
    """Exact weighted rate: blocks of tied scores from the top, the block that crosses `rate` fired fractionally."""
    f = np.zeros(len(s)); ok = np.isfinite(s)
    if not ok.any():
        return f
    vals, inv = np.unique(-s[ok], return_inverse=True); bw = np.bincount(inv, weights=w[ok]); cum = np.cumsum(bw)
    j = int(np.searchsorted(cum, rate + 1e-12, side="right"))
    fb = np.zeros(len(vals)); fb[:j] = 1
    if j < len(vals):
        fb[j] = (rate - (cum[j - 1] if j else 0.0)) / bw[j]
    f[ok] = fb[inv]; return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args(); frozen_committed()
    Z = np.load(FZ / "heads.npz"); THR = json.loads((FZ / "thresholds.json").read_text())
    X, P, U, s0, s16 = load_split(a.data, "test")
    D = Path(a.data); A = {arm: {(r["input"], r["r"]): r for r in rows(D / f"cont-test-{arm}.jsonl")} for arm in "PT"}
    mon = {d["input"]: d for d in rows(OUT / "monitor-test.jsonl")}
    fexp, frisk = head_fn(Z, "exposure"), head_fn(Z, "risk"); f16 = head_fn(Z, "comply16") if "comply16.li" in Z.files else None
    R, dropped = [], Counter()
    for x in X:
        ck = P[x["input"]]["checkpoint"]
        for r in range(MAN["rollouts"]):
            u = U.get(x["input"], {}).get(r); p = A["P"].get((x["input"], r)); t = A["T"].get((x["input"], r))
            if u is None or u.get("error"):
                dropped["U error or missing"] += 1; continue
            if ck and (p is None or p.get("error")):
                dropped["P error or missing"] += 1; continue
            t_ok = bool(ck and t is not None and not t.get("not_applicable"))
            if t_ok and t.get("error"):
                dropped["T error"] += 1; continue
            out = lambda e: (float(bool(e.get("security"))) if x["injection_task"] else 0.0, float(bool(e.get("utility"))), float((e.get("parse_failures") or 0) > 0), float(e.get("llm_seconds") or 0))
            ou = out(u); op = out(p) if ck else ou; ot = out(t) if t_ok else ou
            R.append(dict(input=x["input"], suite=x["suite"], side="attack" if x["injection_task"] else "clean", r=r, ck=bool(ck), t_ok=t_ok, U=ou, P=op, T=ot,
                          s_exposure=float(fexp(s0[x["input"]][None])[0]) if ck and x["input"] in s0 else -np.inf,
                          s_risk=float(frisk(s0[x["input"]][None])[0]) if ck and x["input"] in s0 else -np.inf,
                          s_monitor=float(mon[x["input"]]["score"]) if ck and mon.get(x["input"], {}).get("score") is not None else -np.inf,
                          s_comply16=float(f16(s16[(x["input"], r)][None])[0]) if f16 and t_ok and (x["input"], r) in s16 else -np.inf,
                          exposed=bool(ck and ck["exposed_step"])))
    n = len(R); side = np.array([r["side"] for r in R]); inp = np.array([r["input"] for r in R])
    O = {arm: np.array([r[arm] for r in R]) for arm in "UPT"}   # columns: harmful, utility, parse failure, llm seconds
    S = {g: np.array([r[f"s_{g}"] for r in R]) for g in ("exposure", "risk", "monitor", "comply16")}
    ck = np.array([r["ck"] for r in R]); tok = np.array([r["t_ok"] for r in R])
    groups = {}
    for i, k in enumerate(inp):
        groups.setdefault(k, []).append(i)

    def weights(idx):
        sd = side[idx]; nn = {s: max(1, (sd == s).sum()) for s in W}; return np.array([W[s] / nn[s] for s in sd])

    def policies(idx, rate=RATE):
        w = weights(idx); pol = {"none": (np.zeros(len(idx)), "P"), "always": (ck[idx].astype(float), "P"), "always_T": (tok[idx].astype(float), "T")}
        for g in PROMPT_GATES + ("comply16",):
            s = S[g][idx]; arm = "T" if g == "comply16" else "P"
            if g in THR:
                pol[f"{g}|frozen"] = ((s >= THR[g]) & np.isfinite(s)).astype(float), arm
            pol[f"{g}|replay"] = replay(s, w, rate), arm
        for mode in ("frozen", "replay"):   # random within checkpoint, matched to comply16's count per checkpoint
            if f"comply16|{mode}" not in pol:
                continue
            fc = pol[f"comply16|{mode}"][0]; fr = np.zeros(len(idx)); pos = {}
            for j, i in enumerate(idx):
                pos.setdefault(inp[i], []).append(j)
            for js in pos.values():
                js = [j for j in js if tok[idx[j]]]
                if js:
                    fr[js] = fc[js].sum() / len(js)
            pol[f"random16|{mode}"] = fr, "T"
        return pol, w

    def metrics(idx, pol, w):
        out = {}; att = side[idx] == "attack"; cl = ~att; Ui = O["U"][idx]
        for name, (f, arm) in pol.items():
            Ai = O[arm][idx]; o = f[:, None] * Ai + (1 - f[:, None]) * Ui
            prev = float((f * ((Ui[:, 0] == 1) & (Ai[:, 0] == 0))).sum()); ind = float((f * ((Ui[:, 0] == 0) & (Ai[:, 0] == 1))).sum())
            out[name] = dict(rate_pooled=float((w * f).sum()), rate_attack=float(f[att].mean()) if att.any() else 0.0, rate_clean=float(f[cl].mean()) if cl.any() else 0.0,
                             reminders=float(f.sum()), asr=float(o[att, 0].mean()), clean_completion=float(o[cl, 1].mean()), attack_utility=float(o[att, 1].mean()),
                             prevented=prev, induced=ind, net_per_100=100 * (prev - ind) / f.sum() if f.sum() > 0 else None,
                             parse_failure_episodes=float(o[:, 2].mean()), llm_seconds=float(o[:, 3].mean()))
        return out

    allidx = np.arange(n); pol, w = policies(allidx); M = metrics(allidx, pol, w)
    CON = [("primary: risk - exposure (replay)", "risk|replay", "exposure|replay"), ("risk - exposure (frozen)", "risk|frozen", "exposure|frozen"),
           ("risk - none (replay)", "risk|replay", "none"), ("exposure - none (replay)", "exposure|replay", "none"), ("monitor - risk (replay)", "monitor|replay", "risk|replay"),
           ("comply16 - random16 (replay)", "comply16|replay", "random16|replay"), ("comply16 - risk (replay)", "comply16|replay", "risk|replay"),
           ("risk (replay) - always", "risk|replay", "always"), ("always - none", "always", "none")]
    CON = [c for c in CON if c[1] in M and c[2] in M]
    rng = np.random.default_rng(0); att_in = sorted({inp[i] for i in allidx if side[i] == "attack"}); cl_in = sorted({inp[i] for i in allidx if side[i] == "clean"})
    B = {c[0]: dict(asr=[], clean_completion=[]) for c in CON}; BP = {k: [] for k in M}
    for _ in range(a.draws):
        pick = list(rng.choice(att_in, len(att_in))) + list(rng.choice(cl_in, len(cl_in)))
        idx = np.concatenate([groups[k] for k in pick]); pb, wb = policies(idx); mb = metrics(idx, pb, wb)
        for lab, x1, x2 in CON:
            for k in ("asr", "clean_completion"):
                B[lab][k].append(mb[x1][k] - mb[x2][k])
        for k in M:
            BP[k].append(mb[k]["asr"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    contrasts = {lab: {k: dict(point=M[x1][k] - M[x2][k], ci=ci(B[lab][k])) for k in ("asr", "clean_completion")} for lab, x1, x2 in CON}
    for k in M:
        M[k]["asr_ci"] = ci(BP[k])
    attk = [k for k in groups if side[groups[k][0]] == "attack"]
    mixed = sum(1 for k in attk if ck[groups[k][0]] and 0 < O["U"][groups[k], 0].sum() < len(groups[k]))
    ms_tok = [c["seconds"] / max(1, c["completion_tokens"]) for e in U.values() for u in e.values() for c in (u.get("calls") or []) if c.get("checkpoint")]
    desc = dict(runs=n, dropped=dict(dropped), inputs=dict(attack=len(attk), clean=len(groups) - len(attk)),
                runs_by_side=dict(Counter(side)), checkpoints_attack=int(sum(ck[groups[k][0]] for k in attk)),
                no_checkpoint_inputs=int(sum(not ck[g[0]] for g in groups.values())), mixed_checkpoints=mixed,
                exposed_attack_checkpoints=int(sum(R[groups[k][0]]["exposed"] for k in attk)), t_not_applicable_runs=int((ck & ~tok).sum()),
                untreated_asr_by_suite={s: float(np.mean([O["U"][i, 0] for i in range(n) if R[i]["suite"] == s and side[i] == "attack"])) for s in sorted({r["suite"] for r in R})},
                harmful_untreated_runs=int(O["U"][side == "attack", 0].sum()), attack_runs=int((side == "attack").sum()), clean_runs=int((side == "clean").sum()),
                decision_delay=dict(prompt_gates="0 decode steps", comply16_s=16 * float(np.median(ms_tok)) if ms_tok else None,
                                    monitor_s=float(np.median([d["latency_s"] for d in mon.values()])) if mon else None),
                thresholds=THR)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(dict(meta=desc, policies=M, contrasts=contrasts), indent=1))
    with open(OUT / "units.jsonl", "w") as f:
        for i, r in enumerate(R):
            f.write(json.dumps(dict(input=r["input"], r=r["r"], side=r["side"], ck=r["ck"], t_ok=r["t_ok"], U=r["U"], P=r["P"], T=r["T"],
                                    **{g: (None if not np.isfinite(S[g][i]) else round(float(S[g][i]), 5)) for g in S},
                                    **{k: round(float(v[0][i]), 4) for k, v in pol.items()})) + "\n")
    na, nc = desc["attack_runs"], desc["clean_runs"]
    L = [f"E125 test: {desc['inputs']['attack']} attacked inputs ({na} runs, {desc['harmful_untreated_runs']} harmful untreated), "
         f"{desc['inputs']['clean']} clean ({nc} runs); mixed checkpoints {mixed}; dropped {dict(dropped)}. 95 % CI, {a.draws} draws of whole inputs.", "",
         "| Policy | reminders attack / clean (pooled, weighted) | episode attack success (harmful / attack runs) | clean completion (/ clean runs) | prevented / induced | net avoided per 100 reminders | parse-failure episodes |",
         "|---|---|---|---|---|---|---|"]
    for k, m in M.items():
        net = "" if m["net_per_100"] is None else f"{m['net_per_100']:+.1f}"
        L.append(f"| {k} | {m['rate_attack']:.3f} / {m['rate_clean']:.3f} ({m['rate_pooled']:.3f}) | {m['asr']:.3f} ({m['asr'] * na:.1f} / {na}) "
                 f"[{m['asr_ci'][0]:.3f}, {m['asr_ci'][1]:.3f}] | {m['clean_completion']:.3f} ({m['clean_completion'] * nc:.1f} / {nc}) | "
                 f"{m['prevented']:.1f} / {m['induced']:.1f} | {net} | {m['parse_failure_episodes']:.3f} |")
    L += ["", "| Paired contrast | attack success [95 % CI] | clean completion [95 % CI] |", "|---|---|---|"]
    for lab, c in contrasts.items():
        L.append(f"| {lab} | {c['asr']['point']:+.3f} [{c['asr']['ci'][0]:+.3f}, {c['asr']['ci'][1]:+.3f}] | "
                 f"{c['clean_completion']['point']:+.3f} [{c['clean_completion']['ci'][0]:+.3f}, {c['clean_completion']['ci'][1]:+.3f}] |")
    (OUT / "table.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    plot(allidx, policies, metrics, M)


def plot(allidx, policies, metrics, M):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); col = dict(exposure="#7a7a7a", risk="#c0392b", monitor="#2471a3", comply16="#7d3c98")
    rates = [0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
    sweep = {g: [] for g in col}
    for r in rates:
        pb, wb = policies(allidx, r); mb = metrics(allidx, pb, wb)
        for g in col:
            if f"{g}|replay" in mb:
                sweep[g].append((mb[f"{g}|replay"]["rate_pooled"], mb[f"{g}|replay"]["asr"], mb[f"{g}|replay"]["clean_completion"]))
    for j, key in enumerate(("asr", "clean_completion")):
        for g, pts in sweep.items():
            if pts:
                ax[j].plot([p[0] for p in pts], [p[1 + j] for p in pts], "-", color=col[g], lw=1.2, alpha=0.6, label=f"{g} (budget sweep)")
            for mode, mk in (("replay", "o"), ("frozen", "s")):
                k = f"{g}|{mode}"
                if k in M:
                    ax[j].plot(M[k]["rate_pooled"], M[k][key], mk, color=col[g], mfc=col[g] if mode == "replay" else "white", ms=7)
        for k, mk in (("none", "x"), ("always", "^"), ("random16|replay", "d")):
            if k in M:
                ax[j].plot(M[k]["rate_pooled"], M[k][key], mk, color="black", ms=7, label=k if j == 0 else None)
        ax[j].axvline(RATE, color="#bbbbbb", ls=":", lw=1)
        ax[j].set_xlabel("realised reminder rate (weighted pooled: attack 0.25 / clean 0.75)")
    ax[0].set_ylabel("episode attack success (test attack runs)"); ax[1].set_ylabel("clean task completion (test clean runs)")
    ax[0].set_title("Attack success vs reminder rate"); ax[1].set_title("Clean completion at the same operating points")
    ax[0].legend(fontsize=7, frameon=False); fig.text(0.5, -0.02, "filled circle: exact 10 % replay; open square: frozen dev threshold", ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"fig_e125.{ext}", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
