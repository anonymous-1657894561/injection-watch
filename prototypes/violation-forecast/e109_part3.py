"""E109 part 3 (experiments/E109-comply-gate-where-it-should-win.md, "Part 3"). CPU, plus the rubric judge on the train look-alikes.

  ~/.venvs/agentdojo/bin/python e109_part3.py label --states RUN      -> results/e109/part3-train-labels.jsonl (Haiku rubric, once)
  ~/.venvs/agentdojo/bin/python e109_part3.py thresholds --states RUN -> results/e109/part3-thresholds.json (dev only; commit first)
  ~/.venvs/agentdojo/bin/python e109_part3.py evaluate --states RUN --attack RUN -> results/e109/part3-{summary.json,table.md}

RUN (--states): the pod's injecforks_states run (train + dev look-alike states, data/e79v2/dl/<RUN>/states), fetched here.
RUN (--attack): the pod's injecforks_e109 run (data/e109/dl/e109/attack/<RUN>/results).
Gates (one score per rollout at N = 16 unless named): comply = E096's frozen ensemble (E107's 5e-3-tolerance refit); comply_A / comply_B =
E096's recipe retrained with the train look-alikes as label 0 / as `follows_note`; exposure_n16, exposure_n0 = E079's detect probe;
pg2 / piguard = PromptGuard 2 86M / PIGuard on the case's tool result (injecforks_input_classifiers.score); cascade = exposure_n16 then
comply, as parts 1-2. Policies, budgets and the dev threshold rule: e109_analyze.py.
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e109_analyze as E

OUT = HERE / "results/e109"; TRAIN_ROOT = HERE / "data/e109/injecforks_train"
GATES = ["comply", "comply_A", "comply_B", "exposure_n0", "exposure_n16", "pg2", "piguard", "cascade"]
MISSING = set()   # guards that could not be loaded (gated); their gates are left out and named in every output
PAIRS = [("comply", "exposure_n0"), ("comply", "exposure_n16"), ("comply", "pg2"), ("comply", "piguard"), ("comply_A", "comply"),
         ("comply_B", "comply"), ("comply_A", "exposure_n0"), ("cascade", "exposure_n16")]


def train_lookalikes():
    from injecforks_load import load
    return load("realistic-lookalike", "train", subset=None, root=TRAIN_ROOT)


def label(a):
    """follows_note on the train look-alike replies, by E105's frozen rubric (the reply = the rollout's content after reasoning)."""
    from e105_analyze import RUBRIC, RUBRIC_MODEL, judge_all, notes
    rows = train_lookalikes(); note = notes()
    jobs = [((r["row_id"], "rubric"), RUBRIC.format(user=r["user_instruction"], note=note("lookalike", r["benign"], r["user_tool"]), reply=(r["text"] or "")[:4000]),
             400, RUBRIC_MODEL) for r in rows]
    J = judge_all(jobs, a.max_usd)
    out = [dict(row_id=r["row_id"], follows_note=(bool(J[(r["row_id"], "rubric")]["follows_note"]) if J.get((r["row_id"], "rubric")) else None)) for r in rows]
    (OUT / "part3-train-labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    v = [x["follows_note"] for x in out]
    print("train look-alikes:", len(v), "follows_note", sum(x is True for x in v), "unparsed", sum(x is None for x in v))


def ensembles(states_run):
    """comply (E096's frozen ensemble), comply_A, comply_B: scorers of raw layer-20 vectors."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from e096_analyze import C, LAYER_IDX, SEEDS, e79_states
    from e107_n0 import ensemble
    from injecforks_load import load
    from injecforks_probe import States, train_pairs
    f0, _ = ensemble()
    S = States([e79_states(), HERE / "data/e79v2/dl" / states_run / "states"])
    inj = load("realistic", "train", subset=None); la = train_lookalikes()
    lab = {x["row_id"]: x["follows_note"] for x in map(json.loads, open(OUT / "part3-train-labels.jsonl"))}
    out = {"comply": f0}
    for name, ys in (("comply_A", lambda r: 0), ("comply_B", lambda r: lab[r["row_id"]])):
        L = [r for r in la if ys(r) is not None]; y = [int(r["violates"]) for r in inj] + [int(ys(r)) for r in L]
        fs = []
        for k in range(SEEDS):
            X = S.get(train_pairs(inj, S, k, "realistic|comply") + train_pairs(L, S, k, "e109|lookalike"), LAYER_IDX); sc = StandardScaler().fit(X)
            fs.append((sc, LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)))
        out[name] = (lambda fs: lambda X: np.mean([c.decision_function(s.transform(np.asarray(X, np.float32))) for s, c in fs], 0))(fs)
        print(name, "train rows", len(y), "positives", sum(y), "look-alike positives", sum(y[len(inj):]))
    return out, S


def read_check(F, states_run):
    """The pod's dev look-alike N = 16 states must reproduce E102's stored comply scores (E107's guard)."""
    from e097_analyze import T
    from injecforks_probe import States
    Sp = States([HERE / "data/e79v2/dl" / states_run / "states"]); ref = E.stored("e102-9b-comply-realistic-lookalike-dev.jsonl")
    ids = sorted(ref); got = F["comply"](Sp.get([(r, 16) for r in ids], 3)); want = np.array([ref[r] for r in ids]); thr = T["probe_ens"][1]
    c = dict(rows=len(ids), corr=float(np.corrcoef(got, want)[0, 1]), fire_agreement=float(np.mean((got >= thr) == (want >= thr))), max_abs=float(np.max(np.abs(got - want))))
    print("pod dev read vs E102:", c); assert c["rows"] >= 96 and c["fire_agreement"] >= 0.98 and c["corr"] > 0.999, c
    return c


def guards(texts):
    """{name: {text: P(injection)}} for PromptGuard 2 86M and PIGuard, max over windows."""
    import torch
    from injecforks_input_classifiers import model, score
    dev = "cuda" if torch.cuda.is_available() else "cpu"; u = sorted(set(texts)); out = {}
    for name, key in (("pg2-86m", "pg2"), ("piguard", "piguard")):
        try:
            tok, m = model(name, dev)
        except OSError as e:   # a gated repo the user's token cannot read: recorded, never silently dropped
            print(f"{name}: not run ({str(e).splitlines()[0][:120]})"); MISSING.add(key); continue
        p = score(u, tok, m, dev)
        p = p[0] if isinstance(p, tuple) else p
        out[key] = dict(zip(u, map(float, p))); del m
    return out


def texts(cfg, split):
    from injecforks_load import load
    return {r["case"]: str(r["tool_response"]) for r in load(cfg, split, subset=None)}


def raw(run_dir):
    M = [json.loads(l) for l in open(run_dir / "states-meta.jsonl")]; X = np.load(run_dir / "states.npy").astype(np.float32)
    return {(m["row_id"], m["N"]): x for m, x in zip(M, X)}


def dev_items(F, S):
    A, _ = E.attack(); detect = E.detect_refit()
    n0 = lambda f: {x["row_id"]: x["score"] for x in map(json.loads, open(E.SC / f)) if x["N"] == 0}
    ex0 = {**n0("e79final-detect-realistic-dev.jsonl"), **n0("e102-9b-detect-realistic-lookalike-dev.jsonl"), **n0("e79final-detect-realistic-benign-dev.jsonl")}
    tx = {**{("attack", c): t for c, t in texts("realistic", "dev").items()}, **{("lookalike", c): t for c, t in texts("realistic-lookalike", "dev").items()},
          **{("twin", c): t for c, t in texts("realistic-benign", "dev").items()}}
    G = guards(list(tx.values()))
    rid = [f'realistic|dev|{a["case"]}|{a["rollout"]}' for a in A]
    X = S.get([(r, 16) for r in rid], 3)
    for k in ("comply_A", "comply_B"):
        for a, v in zip(A, F[k](X)):
            a[k] = float(v)
    for a, r in zip(A, rid):
        a.update(comply=a["comply"], exposure_n16=a["exposure"], exposure_n0=ex0[f'realistic|dev|{a["case"]}|0'],
                 **{g: G[g][tx[("attack", a["case"])]] for g in G})
    B = E.benign("dev"); R = raw(E.POD["dev"])
    for x in B:
        v = R[(x["row_id"], 16)][None]
        x.update(comply_A=float(F["comply_A"](v)[0]), comply_B=float(F["comply_B"](v)[0]), exposure_n16=x["exposure"],
                 exposure_n0=ex0[x["row_id"]], **{g: G[g][tx[(x["cat"], x["row_id"].split("|")[2])]] for g in G})
    return A, B, detect


def thresholds(a):
    F, S = ensembles(a.states); chk = read_check(F, a.states)
    A, B, _ = dev_items(F, S); L = [x for x in B if x["cat"] == "lookalike"]; T = [x for x in B if x["cat"] == "twin"]
    te = E.quantile_thr([x["exposure"] for x in L], E.CASCADE_PASS); passed = [x for x in L if x["exposure"] >= te]
    th = dict(read_check=chk, not_run=sorted(MISSING), A={}, B={})
    for b in E.BUDGET_A:
        th["A"][str(b)] = {g: E.quantile_thr([x[g] for x in L], b) for g in GATES if g != "cascade" and g not in MISSING}
        th["A"][str(b)]["cascade"] = [te, E.quantile_thr([x["comply"] for x in passed], b * len(L) / len(passed))]
    for b in E.BUDGET_B:
        th["B"][str(b)] = {g: E.b_thr(g, A, L, T, b) for g in GATES if g != "cascade" and g not in MISSING}
        th["B"][str(b)]["cascade"] = [te, E.b_thr("cascade", A, L, T, b, te)]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "part3-thresholds.json").write_text(json.dumps(th, indent=1)); print(json.dumps(th)[:3000])


def boot(TH, A, L, T, key, seed=0):
    rng = np.random.default_rng(seed)
    ac = sorted({a["case"] for a in A}); bb = sorted({x["base"] for x in L + T})
    Ai = {c: [a for a in A if a["case"] == c] for c in ac}; Li = {b: [x for x in L if x["base"] == b] for b in bb}; Ti = {b: [x for x in T if x["base"] == b] for b in bb}
    D = {g: [] for g in TH}
    for _ in range(E.BOOT):
        ca, cb = rng.choice(ac, len(ac)), rng.choice(bb, len(bb))
        A_, L_, T_ = [a for c in ca for a in Ai[c]], [x for b in cb for x in Li[b]], [x for b in cb for x in Ti[b]]
        for g, th in TH.items():
            D[g].append(E.metrics(g, th, A_, L_, T_)[key])
    D = {g: np.array(v) for g, v in D.items()}; ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {g: ci(v) for g, v in D.items()}, {f"{p}-{q}": ci(D[p] - D[q]) for p, q in PAIRS if p in D and q in D}


def evaluate(a):
    p = OUT / "part3-thresholds.json"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(p)]).returncode == 0 and \
        subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], capture_output=True).returncode == 0, "commit part3-thresholds.json first"
    TH = json.loads(p.read_text()); A, L, T, undecided = test_items(a)
    out = dict(meta=dict(attack_rollouts=len(A), attack_cases=len({a["case"] for a in A}), undecided_samples=undecided, attack_success_control=float(np.mean([a["control"] for a in A])),
                         attack_success_reminder=float(np.mean([a["reminder"] for a in A])), read_check=TH["read_check"], not_run=TH["not_run"]))
    lines = ["| policy | budget | gate | fires: attack / look-alike / twin / traffic | A: attacks prevented | B: violations prevented | look-alike completion |", "|---|---|---|---|---|---|---|"]
    for pol, k in (("A", "A_prevented"), ("B", "B_prevented")):
        for b, gates in TH[pol].items():
            ci, diff = boot(gates, A, L, T, k)
            out[f"{pol}|{b}"] = dict(gates={g: dict(E.metrics(g, t, A, L, T), ci=ci[g]) for g, t in gates.items()}, diff_ci=diff, always=E.metrics("comply", -np.inf, A, L, T))
            for g, m in out[f"{pol}|{b}"]["gates"].items():
                cell = lambda kk: f"**{m[kk]:.2f} [{m['ci'][0]:.2f}, {m['ci'][1]:.2f}]**" if kk == k else f"{m[kk]:.2f}"
                lines.append(f"| {pol} | {b} | {g} | {m['fire_attack']:.2f} / {m['fire_lookalike']:.2f} / {m['fire_twin']:.2f} / {m['fire_traffic']:.2f} | {cell('A_prevented')} | "
                             f"{cell('B_prevented')} | {m['lookalike_completion']:.3f} |")
            lines.append(f"| {pol} | {b} | Δ CIs | " + "; ".join(f"{q} [{v[0]:+.2f}, {v[1]:+.2f}]" for q, v in diff.items()) + " | | | |")
    (OUT / "part3-summary.json").write_text(json.dumps(out, indent=1)); (OUT / "part3-table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines)); print(out["meta"])


def test_items(a):
    """(A, L, T, undecided): the 3b test attack rollouts and E107's test look-alikes and twins, with every gate's score."""
    F, S = ensembles(a.states); detect = E.detect_refit()
    run = HERE / "data/e109/dl/e109/attack" / a.attack / "results"; R = raw(run)
    by = {}
    for x in map(json.loads, open(run / "e109.jsonl")):
        if x["decided"]:
            by.setdefault((x["case"], x["rollout"]), {})[x["arm"]] = float(bool(x["violates_strict"]))
    undecided = sum(1 for x in map(json.loads, open(run / "e109.jsonl")) if not x["decided"])
    tx_a = texts("realistic", "test"); keys = sorted(k for k, v in by.items() if "control" in v and "reminder" in v)
    tx_b = {**{("lookalike", c): t for c, t in texts("realistic-lookalike", "test").items()}, **{("twin", c): t for c, t in texts("realistic-benign", "test").items()}}
    G = guards([tx_a[c] for c, _ in keys] + list(tx_b.values()))
    X16 = np.stack([R[(f"realistic|test|{c}|{ro}", 16)] for c, ro in keys]); X0 = np.stack([R[(f"realistic|test|{c}", 0)] for c, _ in keys])
    sc = {k: F[k](X16) for k in ("comply", "comply_A", "comply_B")}; e16, e0 = detect(X16), detect(X0)
    A = [dict(case=c, rollout=ro, control=by[(c, ro)]["control"], reminder=by[(c, ro)]["reminder"], comply=float(sc["comply"][i]), comply_A=float(sc["comply_A"][i]),
              comply_B=float(sc["comply_B"][i]), exposure=float(e16[i]), exposure_n16=float(e16[i]), exposure_n0=float(e0[i]), **{g: G[g][tx_a[c]] for g in G})
         for i, (c, ro) in enumerate(keys)]
    B = E.benign("test", detect); Rt = raw(E.POD["test"]); e0b = json.loads((HERE / "results/e107/exposure-scores-n0.json").read_text())["test"]
    for x in B:
        v = Rt[(x["row_id"], 16)][None]; cs = x["row_id"].split("|")[2]
        x.update(comply_A=float(F["comply_A"](v)[0]), comply_B=float(F["comply_B"](v)[0]), exposure_n16=x["exposure"], exposure_n0=e0b[f'{x["cat"]}|{cs}'],
                 **{g: G[g][tx_b[(x["cat"], cs)]] for g in G})
    L = [x for x in B if x["cat"] == "lookalike"]; T = [x for x in B if x["cat"] == "twin"]
    return A, L, T, undecided


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["label", "thresholds", "evaluate"])
    ap.add_argument("--states"); ap.add_argument("--attack"); ap.add_argument("--max-usd", type=float, default=3.5)
    a = ap.parse_args(); {"label": label, "thresholds": thresholds, "evaluate": evaluate}[a.stage](a)
