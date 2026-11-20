"""Dollars per monitor prediction (one call = one row at one N), with prompt caching as billed and without it (every prompt token
at the full input price), from the stored per-call usage. CPU, free. Prices: OpenRouter, 2026-11-03, $ per token; the same for
both models. The cached formula is checked against OpenRouter's own `cost` on every live call before anything is reported.

  python injecforks_monitor_cost.py   -> results/injecforks/analysis/monitor-cost-dev.json
"""
import json, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE / "results/injecforks/monitor"
PRICE = dict(prompt=2e-6, completion=10e-6, cache_read=0.1e-6, cache_write=2.5e-6)
RUNS = {"gpt-6.1-sol low": "dev-gpt61sol-low-v1", "sonnet-5.5 low": "dev-sonnet55-v1"}


def cost(u, cache):
    d = u.get("prompt_tokens_details") or {}
    p, c = u["prompt_tokens"], u["completion_tokens"]
    if not cache:
        return p * PRICE["prompt"] + c * PRICE["completion"]
    r, w = d.get("cached_tokens") or 0, d.get("cache_write_tokens") or 0
    return r * PRICE["cache_read"] + w * PRICE["cache_write"] + (p - r - w) * PRICE["prompt"] + c * PRICE["completion"]


out = {}
for name, tag in RUNS.items():
    live = [d for d in map(json.loads, open(M / f"{tag}.jsonl")) if "batch" not in d and d["usage"].get("prompt_tokens")]
    unbilled = [d for d in live if not float(d["usage"]["cost"])]   # cut-off replies OpenRouter did not charge (2 on Sonnet)
    print(name, len(unbilled), "unbilled calls kept at the formula's price")
    bad = [d for d in live if float(d["usage"]["cost"]) and abs(cost(d["usage"], True) - float(d["usage"]["cost"])) > 1e-6]
    assert not bad, (name, len(bad), "calls where the cached formula disagrees with OpenRouter's cost")
    for grp in ("injected", "benign", "lookalike"):
        rows = [d for d in live if (d["row_id"].split("|")[0].split("-") + ["injected"])[1] == grp]
        if not rows:
            continue
        u = [d["usage"] for d in rows]
        out[f"{name}|{grp}"] = r = dict(
            calls=len(rows), prompt_tokens=round(st.mean(x["prompt_tokens"] for x in u)),
            completion_tokens=round(st.mean(x["completion_tokens"] for x in u), 1),
            cached_share=round(sum((x.get("prompt_tokens_details") or {}).get("cached_tokens") or 0 for x in u) / sum(x["prompt_tokens"] for x in u), 3),
            usd_cached=round(st.mean(cost(x, True) for x in u), 6), usd_uncached=round(st.mean(cost(x, False) for x in u), 6))
(HERE / "results/injecforks/analysis/monitor-cost-dev.json").write_text(json.dumps(out, indent=1))
print("| monitor | rows | calls | prompt tokens | output tokens | cached share | $ / prediction, cached | uncached |")
print("|---|---|---|---|---|---|---|---|")
for k, r in out.items():
    print(f"| {k.replace('|', ' | ')} | {r['calls']} | {r['prompt_tokens']} | {r['completion_tokens']} | {r['cached_share']:.0%} | "
          f"{r['usd_cached']:.5f} | {r['usd_uncached']:.5f} |")
