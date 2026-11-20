"""E104: one table of seconds and dollars per prediction for every method, cached and from scratch. CPU, free.
GPU pieces from the timing pod (results/e104-timing/timing.json, median over its 10 cases), priced at the run's billed $/h
(results/e104-timing/billing.json, from the run's ledger record); the monitor from its timed live sample (results/injecforks/monitor/e104-lat-*.jsonl, billed cost).

  python e104_analyze.py   -> results/e104-timing/summary.md
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = json.load(open(HERE / "results/e104-timing/timing.json"))
USD_H = json.load(open(HERE / "results/e104-timing/billing.json"))["usd_per_hour"]   # copied from the run's ledger record
med = lambda xs: st.median(xs)
C = T["cases"]
unit = med([c["agent_decode_token"] for c in C])   # one generated agent token: the comparison unit
rows = []


def gpu(method, n, cached, uncached):
    rows.append((method, n, cached, uncached, cached * USD_H / 3600 if cached is not None else None, uncached * USD_H / 3600, None))


for n in ("16", "32"):
    g = lambda k: med([c["n"][n][k] for c in C if n in c["n"]])
    gpu("9B probe (comply / detect)", n, med([c["probe_dot"] for c in C]), g("probe_uncached"))
    gpu("9B prefix-only probe", n, None, g("prefixonly_uncached"))
    gpu("9B self-report", n, g("selfreport_cached"), g("selfreport_uncached"))
    gpu("Distill probe (8B)", n, g("distill_cached"), g("distill_uncached"))
for name, by in T["classifiers"].items():
    s = med(list(by.values())); rows.append((name, "any", None, s, None, s * USD_H / 3600, None))
for model, tag in (("monitor: GPT-6.1 Sol low", "gpt61sol"), ("monitor: Sonnet 5.5 low", "sonnet55")):
    arm = {k: [json.loads(l) for l in open(HERE / f"results/injecforks/monitor/e104-lat-{tag}{k}.jsonl")] for k in ("", "-nocache")}
    lat = {k: med([r["latency_s"] for r in v]) for k, v in arm.items()}; usd = {k: st.mean(float(r["usage"]["cost"]) for r in v) for k, v in arm.items()}
    rows.append((model, "16, 32", lat[""], lat["-nocache"], usd[""], usd["-nocache"], "API round trip"))

L = [f"# E104 timing summary\n", f"GPU: {T['gpu']}, bf16, ${USD_H}/h, {len(C)} realistic dev cases (prompt {min(c['prompt_tokens'] for c in C)}–"
     f"{max(c['prompt_tokens'] for c in C)} tokens), median of {T['reps']} repeats. Fast linear-attention kernels: {T['fast_kernels']}.",
     f"Unit: one generated agent token = {unit * 1e3:.1f} ms; prompt prefill (the agent's own) = {med([c['agent_prefill'] for c in C]):.2f} s.\n",
     "| method | N | s, cached | s, from scratch | $, cached | $, from scratch | from scratch, in agent tokens |", "|---|---|---|---|---|---|---|"]
f = lambda x, p: "—" if x is None else (f"{x * 1e3:.2f} ms" if p == "s" and x < 0.1 else f"{x:.3f} s" if p == "s" else f"{x:.6f}")
for m, n, cs, us, cu, uu, note in rows:
    L.append(f"| {m} | {n} | {f(cs, 's')} | {f(us, 's')} | {f(cu, '$')} | {f(uu, '$')} | {note or f'{us / unit:.0f}'} |")
(HERE / "results/e104-timing/summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
