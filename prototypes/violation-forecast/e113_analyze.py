"""E113: E79's realistic test comparison on all 16 rollouts of the 299 eval cases (experiments/E113-e79-test-all-rollouts.md). CPU.

  ~/.venvs/e01/bin/python e113_analyze.py      # -> results/injecforks/analysis/e113.{json,md}

Rows: eval-v1.2 realistic test rows plus their cases' other rollouts (subset evalextra), at N = 8 and 16; every method on identical
rows. Case-mean within-case AUROC, two-way bootstrap (cases x rollout numbers, e79_amended), 2,000 draws. The same numbers on the
eval rows alone are printed beside it. Self-report is included when its evalextra scores exist.
"""
import json
from pathlib import Path

import e79_amended as E
from injecforks_load import load

HERE = Path(__file__).resolve().parent
SC = HERE / "results/injecforks/scores"; AN = HERE / "results/injecforks/analysis"
NS = [8, 16]


def scores(*names):
    s = {}
    for n in names:
        p = SC / n
        if p.exists():
            s.update({(x["row_id"], int(x["N"])): float(x["score"]) for x in map(json.loads, open(p))})
    return s


M = {"probe": scores("e79final-e113-comply-realistic-test.jsonl"),
     "monitor": scores("monitor-test-gpt61sol-low-v1-realistic-test.jsonl",
                       "monitor-test-gpt61sol-low-v1-evalextra-realistic-test-evalextra.jsonl")}
sr = scores("selfreport-turn-fp32-realistic-test.jsonl", "selfreport-turn-fp32-realistic-test-evalextra.jsonl")
ev, ex = load("realistic", "test"), load("realistic", "test", subset="evalextra")
if any(k[0] in {r["row_id"] for r in ex} for k in sr):
    M["selfreport"] = sr
fmt = lambda s: f"{s['est']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]"
out, md = {}, ["# E113: realistic test, all 16 rollouts of the 299 eval cases (two-way bootstrap, 2,000 draws)", ""]
for name, rows in (("all16", ev + ex), ("eval8", ev)):
    md += [f"## {name}", "", "| N | rows | cases | " + " | ".join(M) + " | " + " | ".join(f"probe − {b}" for b in M if b != "probe") + " |",
           "|---" * (3 + 2 * len(M) - 1) + "|"]
    out[name] = {}
    D = {}
    for n in NS:
        T, missing = E.table(rows, n, M)
        D[n] = E.draws(T, E.weights(T["case"], T["roll"])); E.assert_point(T, D[n])
        r = dict(rows=len(T["rows"]), missing=missing, cases=len(set(T["case"])),
                 auroc={m: E.summ(v) for m, v in D[n].items()},
                 delta={b: E.summ(D[n]["probe"] - D[n][b]) for b in M if b != "probe"})
        out[name][n] = r
        md.append(f"| {n} | {r['rows']} | {r['cases']} | " + " | ".join(f"{r['auroc'][m]['est']:.3f}" for m in M) + " | "
                  + " | ".join(fmt(r["delta"][b]) for b in r["delta"]) + " |")
    early = {b: E.summ((D[8]["probe"] - D[8][b] + D[16]["probe"] - D[16][b]) / 2) for b in M if b != "probe"}
    out[name]["mean_8_16"] = early
    md += [f"| 8+16 mean | | | " + " | " * (len(M) - 1) + " | " + " | ".join(fmt(early[b]) for b in early) + " |", ""]
AN.mkdir(parents=True, exist_ok=True)
(AN / "e113.json").write_text(json.dumps(out, indent=1))
(AN / "e113.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
