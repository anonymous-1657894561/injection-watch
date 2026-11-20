"""E125 pod helpers (experiments/E125-agentdojo-gptoss-heldout.md): artifact repo check and upload, generation smoke check, pod summary.

  python e125_pod_util.py check-repo
  python e125_pod_util.py upload --data DIR --run RUN_ID        # DIR -> <artifact_repo>/e125/<RUN_ID>/ (private repo only)
  python e125_pod_util.py smoke-check --data DIR
  python e125_pod_util.py summary --data DIR --out FILE         # counts, errors, parse failures; outcome rates for dev only
"""
import argparse, json
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "configs/e125.yaml").read_text())


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def check_repo():
    from huggingface_hub import HfApi
    api = HfApi()
    if not api.repo_info(CFG["artifact_repo"], repo_type="model").private:
        raise SystemExit("artifact repo must be private")
    print("artifact repo private:", CFG["artifact_repo"])


def upload(data, run):
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=CFG["artifact_repo"], folder_path=data, path_in_repo=f"e125/{run}", allow_patterns=["*.jsonl", "states/*"])
    print("uploaded", data, "->", f"{CFG['artifact_repo']}/e125/{run}")


def smoke_check(data):
    P = rows(Path(data) / "prefix-dev.jsonl"); C = {a: rows(Path(data) / f"cont-dev-{a}.jsonl") for a in "UPT"}
    print(len(P), "prefixes; checkpoints", sum(bool(p["checkpoint"]) for p in P), "errors", [p["error"] for p in P if p["error"]])
    for a, R in C.items():
        print(a, len(R), "rows; errors", [r.get("error") for r in R if r.get("error")], "not applicable", sum(bool(r.get("not_applicable")) for r in R),
              "k_ids", Counter((r.get("k_ids") or {}).get("source") for r in R))
    assert P and not any(p["error"] for p in P) and any(p["checkpoint"] for p in P), "smoke prefixes"
    assert all(C[a] for a in "UP") and not any(r.get("error") for a in "UPT" for r in C[a]), "smoke continuations"


def summary(data, out):
    D = Path(data); S = {}
    for split in ("dev", "test"):
        P = rows(D / f"prefix-{split}.jsonl"); s = dict(inputs=len(P), checkpoints=sum(bool(p["checkpoint"]) for p in P),
                                                       prefix_errors=sum(bool(p["error"]) for p in P),
                                                       checkpoint_call=Counter(p["checkpoint"]["k"] for p in P if p["checkpoint"]))
        for a in "UPT":
            R = rows(D / f"cont-{split}-{a}.jsonl")
            if not R:
                continue
            live = [r for r in R if not r.get("not_applicable")]
            s[a] = dict(rows=len(R), errors=sum(bool(r.get("error")) for r in live), not_applicable=len(R) - len(live),
                        episodes_with_parse_failure=sum((r.get("parse_failures") or 0) > 0 for r in live),
                        live_calls=sum(r.get("live_calls") or 0 for r in live), parse_failures=sum(r.get("parse_failures") or 0 for r in live),
                        k_ids_source=Counter((r.get("k_ids") or {}).get("source") for r in live) if a == "U" else None)
            if split == "dev":   # dev outcomes may be read: they train the heads
                att = [r for r in live if r["attacked"] and r.get("security") is not None]; cl = [r for r in live if not r["attacked"] and r.get("utility") is not None]
                s[a].update(attack_success=sum(r["security"] for r in att) / max(1, len(att)), clean_completion=sum(r["utility"] for r in cl) / max(1, len(cl)))
        if split == "dev":
            ck = [p["checkpoint"] for p in P if p["checkpoint"] and p["injection_task"]]
            s["attacked_checkpoint_exposed_step"] = sum(c["exposed_step"] for c in ck) / max(1, len(ck))
        S[split] = s
    Path(out).write_text(json.dumps(S, indent=1, default=str)); print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check-repo", "upload", "smoke-check", "summary"])
    ap.add_argument("--data"); ap.add_argument("--run"); ap.add_argument("--out")
    a = ap.parse_args()
    {"check-repo": lambda: check_repo(), "upload": lambda: upload(a.data, a.run), "smoke-check": lambda: smoke_check(a.data),
     "summary": lambda: summary(a.data, a.out)}[a.cmd]()
