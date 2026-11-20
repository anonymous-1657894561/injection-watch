"""E136, the one pre-registered test read. Nothing here is fitted or selected: the three saved seed models per (config, N)
(e136_final.py) and the combination weights in results/e136/frozen.json.

Scores the test eval-v1.2 rows, both configs, every N with saved models -> scores/e136-{ft,comb}-<cfg>-test.jsonl, plus the
per-seed within-case AUROC (seed spread) -> results/e136/test-seeds.json. Then, by hand (see E136):
  injecforks_score.py --split test --method ft=... probe=... emb=... comb=... length=... --baseline probe --out e136-test
  injecforks_score.py --split test --method ft=... emb=... --baseline emb --out e136-test-vs-emb
"""
import json

from e136_common import NS, OUT, SC, gpu_lock, load, model_path, score_ens
from injecforks_metrics import within_case_mean

FZ = json.loads((OUT / "frozen.json").read_text())
SEEDS = FZ["seeds"]
seeds_out = {}
for cfg in ("smoke", "realistic"):
    rows = load(cfg, "test")
    probe = {(x["row_id"], int(x["N"])): x["score"] for x in map(json.loads, open(SC / f"e79final-comply-{cfg}-test.jsonl"))}
    ft = {(r["row_id"], 0): 0.0 for r in rows}
    for N in NS:
        if not all(model_path(cfg, N, s).exists() for s in SEEDS):
            print("no models for", cfg, N); continue
        assert str(N) in FZ["comb"][cfg]
        with gpu_lock():
            E = score_ens(cfg, N, SEEDS, rows)
        ft.update({(k, N): v[0] for k, v in E.items()})
        b = [r for r in rows if r["row_id"] in E]
        cs, ys = [r["case"] for r in b], [r["violates"] for r in b]
        seeds_out[f"{cfg} N={N}"] = dict(ens=within_case_mean(cs, ys, [E[r["row_id"]][0] for r in b]),
                                         seeds=[within_case_mean(cs, ys, [E[r["row_id"]][1][i] for r in b]) for i in range(len(SEEDS))])
        print(cfg, N, seeds_out[f"{cfg} N={N}"], flush=True)
    comb = {k: FZ["comb"][cfg][str(k[1])]["a"] * probe[k] + FZ["comb"][cfg][str(k[1])]["b"] * v for k, v in ft.items()}
    for nm, S in (("ft", ft), ("comb", comb)):
        with open(SC / f"e136-{nm}-{cfg}-test.jsonl", "w") as fh:
            fh.writelines(json.dumps({"row_id": k[0], "N": k[1], "score": v}) + "\n" for k, v in sorted(S.items(), key=lambda x: (x[0][1], x[0][0])))
(OUT / "test-seeds.json").write_text(json.dumps(seeds_out, indent=1))
