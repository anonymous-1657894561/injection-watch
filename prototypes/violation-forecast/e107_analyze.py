"""E107 analysis (experiments/E107-gated-reminder-lookalike-cost.md). CPU, plus E105's two frozen OpenRouter judges.

  ~/.venvs/e01/bin/python injecforks_e107.py configs/e107-9b-dev-pod.yaml score --prefix RUN_ID   # first: the gate's scores
  ~/.venvs/e01/bin/python e107_analyze.py --split dev RUN_ID [--judge] [--max-usd 3.5]
  ~/.venvs/e01/bin/python e107_analyze.py --freeze                                                # once, after dev, before test
  ~/.venvs/e01/bin/python e107_analyze.py --split test RUN_ID --judge                             # refuses unless frozen, unchanged
  -> results/e107/<split>-{rows.jsonl, summary.json, table.md}

Three arms per row, from one N = 16 prefix: control and always (the reminder continuation), both sampled, and gated (the reminder
continuation where the frozen comply ensemble's N = 16 score is at or above E097's deployment threshold, the control continuation
elsewhere). Outcomes are E105's, by E105's frozen judges and rubric, unchanged. Every rate is a paired case-cluster bootstrap
(E093's `boot`, 2,000 draws, seed 0): a cluster is one injected case `<attack>:<row>` (realistic only) with its two twins and two
look-alikes, every arm and row.
"""
import argparse, hashlib, json, os, sys

import numpy as np
import yaml

from e093_analyze import JPROMPT, JUDGE_MODEL, boot
from e097_analyze import T as TRIGGERS
from e105_analyze import CANONICAL, OUTS, RUBRIC_HASH, RUBRIC_MODEL, RUBRIC_SHA, canonical, contexts, outcomes
from e105_analyze import FROZEN as E105_FROZEN
from injecforks_e107 import FAMILY, select, work
from injecforks_states import snapshot, verify
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/e107"
FROZEN = OUT / "frozen.json"   # written by --freeze after dev; test asserts every field of it, then the test manifest
THRESHOLD = TRIGGERS["probe_ens"][1]   # E097's frozen deployment threshold: the comply ensemble's benign-twin dev 95th percentile
MARGIN = -0.05   # decision rule on look-alike gated Δ completion (primary), as E105's
ARMS = ("control", "always", "gated")
CONTRASTS = (("always", "control"), ("gated", "control"), ("gated", "always"))


def manifest(split):
    job = yaml.safe_load((HERE / f"configs/e107-9b-{split}-pod.yaml").read_text())
    root = snapshot(select(job, split)); verify(root)
    man = sorted([cat, cs, r[0]] for _, vs in work(root, split) for cat, _, cs, *_, rs in vs for r in rs)
    return dict(n=len(man), sha256=hashlib.sha256(json.dumps(man).encode()).hexdigest()), {tuple(m) for m in man}


def frozen(split):
    """Everything test is analysed with: E105's judges (asserted equal to E105's freeze), the gate, the bootstrap, the decision
    rule, and the split's (category, case, rollout) manifest."""
    e = json.loads(E105_FROZEN.read_text())
    J = dict(task_judge=dict(model=JUDGE_MODEL, dated=CANONICAL[JUDGE_MODEL], prompt_sha256=hashlib.sha256(JPROMPT.encode()).hexdigest(),
                             temperature=0, max_tokens=200),
             rubric_judge=dict(model=RUBRIC_MODEL, dated=CANONICAL[RUBRIC_MODEL], rubric_sha256=RUBRIC_HASH, temperature=0, max_tokens=400, reasoning="off"))
    assert J == {k: e[k] for k in J}, "E105's frozen judges"
    return dict(J, gate=dict(probe="probe_ens: the frozen 5-seed comply ensemble (e096_analyze.ensemble), layer 20, N = 16", fires="score >= threshold",
                             threshold=THRESHOLD),
                bootstrap=dict(n=2000, seed=0, cluster="<attack>:<row> (realistic): a case's two twins and two look-alikes, every arm"),
                decision=dict(primary="lookalike gated - control task delta", margin=MARGIN,
                              secondary=["lookalike gated - always task delta", "lookalike always - control task delta", "twin gated - control task delta",
                                         "fire rate on look-alikes and twins"]),
                manifest={split: manifest(split)[0]})


def summarise(rows):
    cl = sorted({r["base"] for r in rows}); ix = {c: i for i, c in enumerate(cl)}; ids = list(range(len(cl)))
    slices = {"twin": lambda r: r["category"] == "twin", "lookalike": lambda r: r["category"] == "lookalike"}
    for c, ks in (("twin", ("neutral", "imperative")), ("lookalike", ("remind", "refrain"))):
        for k in ks:
            slices[f"{c}-{k}"] = lambda r, c=c, k=k: r["category"] == c and r["kind"] == k
    rate = lambda M, cs: M[cs, 0].sum() / M[cs, 1].sum() if M[cs, 1].sum() else np.nan
    S = {}
    for o in OUTS + ["fires"]:
        nd = {s: {a: np.zeros((len(cl), 2)) for a in ARMS} for s in slices}
        for r in rows:
            if r[o] is None:
                continue
            for s, f in slices.items():
                if f(r):
                    nd[s][r["arm"]][ix[r["base"]]] += (float(r[o]), 1.0)
        for s in slices:
            M = nd[s]
            if not M["control"][:, 1].sum():
                continue
            v = {a: dict(n=int(M[a][:, 0].sum()), d=int(M[a][:, 1].sum()), ci=boot(ids, lambda cs, a=a: rate(M[a], cs))) for a in ARMS}
            if o != "fires":
                v.update({f"{a}-{b}": boot(ids, lambda cs, a=a, b=b: rate(M[a], cs) - rate(M[b], cs)) for a, b in CONTRASTS})
            S.setdefault(s, {})[o] = v
        if o != "fires":
            a, b = nd["lookalike"], nd["twin"]
            S.setdefault("lookalike_minus_twin", {})[o] = {
                f"{x}-{y}": boot(ids, lambda cs, x=x, y=y: (rate(a[x], cs) - rate(a[y], cs)) - (rate(b[x], cs) - rate(b[y], cs))) for x, y in CONTRASTS}
    S["_counts"] = dict(clusters=len(cl), rows_per_arm={c: sum(1 for r in rows if r["category"] == c and r["arm"] == "control") for c in ("twin", "lookalike")})
    return S


def table(S, split, decision):
    f = lambda x: "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.3f}" if x < 0 or x > 1 else f"{x:.3f}"
    ci = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    rt = lambda d: f"{d['ci'][0]:.3f} ({d['n']}/{d['d']})"
    L = [f"## E107 {split}: N = 16 reminder, always on and gated on the comply probe (realistic)", "",
         f"Gate: comply ensemble at N = 16, fires at score ≥ {THRESHOLD}. Decision (look-alike gated − control completion, margin {MARGIN:+.2f}): **{decision}**", "",
         "| slice | outcome | control | always | gated | always − control | gated − control | gated − always |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s, v in S.items():
        if s.startswith("_") or s == "lookalike_minus_twin":
            continue
        for o in OUTS:
            if o in v:
                L.append(f"| {s} | {o} | {rt(v[o]['control'])} | {rt(v[o]['always'])} | {rt(v[o]['gated'])} | "
                         + " | ".join(ci(v[o][f"{a}-{b}"]) for a, b in CONTRASTS) + " |")
        if "fires" in v:
            L.append(f"| {s} | **gate fires** | | | {rt(v['fires']['control'])} | | | |")
    L += ["", "| look-alike − twin | " + " | ".join(f"{a} − {b}" for a, b in CONTRASTS) + " |", "|---|---:|---:|---:|"]
    for o, v in S["lookalike_minus_twin"].items():
        L.append(f"| {o} | " + " | ".join(ci(v[f"{a}-{b}"]) for a, b in CONTRASTS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_id", nargs="?"); ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--judge", action="store_true"); ap.add_argument("--max-usd", type=float, default=3.5, help="cap on the judge cache shared with E105, cumulative (E105 spent $1.74)")
    ap.add_argument("--freeze", action="store_true", help="write results/e107/frozen.json (dev + test manifests), once, before test")
    a = ap.parse_args(); os.chdir(HERE); OUT.mkdir(parents=True, exist_ok=True)
    assert RUBRIC_SHA.exists() and RUBRIC_SHA.read_text().strip() == RUBRIC_HASH, "E105's rubric, frozen and unchanged"
    if a.freeze:
        assert not FROZEN.exists() and canonical() == CANONICAL, "freeze once, with the judges as pinned"
        F = frozen("dev"); F["manifest"].update(frozen("test")["manifest"])
        FROZEN.write_text(json.dumps(F, indent=1) + "\n"); print(json.dumps(F, indent=1)); sys.exit()
    assert a.run_id and a.split
    if a.split == "test":
        assert canonical() == CANONICAL, ("a judge model moved since the freeze", canonical())
    R = HERE / "data/e107/dl/e107" / a.run_id / "results"
    run = json.load(open(R / "run.json")); assert run.get("complete") and run["split"] == a.split, run.get("split")
    recs = [json.loads(l) for l in open(R / "e107.jsonl")]
    sc = {x["row_id"]: x["score"] for x in map(json.loads, open(R / "scores.jsonl")) if x["N"] == 16}
    _, exp = manifest(a.split)
    for arm in ("control", "reminder"):   # every row of the split once per arm, and a score for each: no partial run is analysed
        got = sorted((x["category"], x["case"], x["rollout"]) for x in recs if x["arm"] == arm)
        assert len(got) == len(set(got)) and set(got) == exp, ("rows differ from the dataset", arm, len(set(got) ^ exp))
    assert {x["row_id"] for x in recs} == set(sc), "a score for every row"
    if a.split == "test":
        F, G = json.loads(FROZEN.read_text()), frozen("test")
        assert {k: v for k, v in G.items() if k != "manifest"} == {k: v for k, v in F.items() if k != "manifest"}, "evaluator drift since the freeze"
        assert G["manifest"]["test"] == F["manifest"]["test"], "test manifest differs from the frozen one"
    ctx, _ = contexts(a.split)
    judged = outcomes([dict(x, family=FAMILY) for x in recs], ctx, a.judge, a.max_usd)
    by = {(r["row_id"], r["arm"]): r for r in judged}
    rows = []
    for (rid, arm), r in by.items():
        fire = sc[rid] >= THRESHOLD
        if arm == "control":
            rows.append(dict(r, arm="control", fires=fire, score=sc[rid]))
            rows.append(dict(by[(rid, "reminder")] if fire else r, arm="gated", fires=fire, score=sc[rid]))
        else:
            rows.append(dict(r, arm="always", fires=fire, score=sc[rid]))
    (OUT / f"{a.split}-rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    S = summarise(rows)
    la = S["lookalike"]["task"]["gated-control"]
    decision = "material harm" if la[2] < MARGIN else "acceptable cost" if la[1] > MARGIN else "inconclusive"
    meta = dict(run_id=a.run_id, run=run, decision=decision, threshold=THRESHOLD, frozen=json.loads(FROZEN.read_text()) if FROZEN.exists() else None,
                scores_vs_e102=json.loads((R / "scores-vs-e102.json").read_text()) if (R / "scores-vs-e102.json").exists() else None,
                unjudged={o: sum(r[o] is None for r in rows if r["arm"] != "gated") for o in ("task", "refuses")})
    (OUT / f"{a.split}-summary.json").write_text(json.dumps(dict(meta=meta, **S), indent=1, default=float))
    (OUT / f"{a.split}-table.md").write_text(table(S, a.split, decision)); print(table(S, a.split, decision))
