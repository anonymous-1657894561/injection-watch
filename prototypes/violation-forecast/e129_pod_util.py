"""E129 pod helpers (experiments/E129-agentdojo-multi-attack.md): artifact repo check and upload, generation smoke check, pod summary.

  python e129_pod_util.py check-repo
  python e129_pod_util.py upload --data DIR --run RUN_ID --model M      # DIR -> <artifact_repo>/e129/<M>/<RUN_ID>/ (private repo only)
  python e129_pod_util.py smoke-check --data DIR --split S
  python e129_pod_util.py summary --data DIR --split S --out FILE      # counts, errors, parse failures; outcome rates for dev only
  python e129_pod_util.py fetch --model M --run RUN_ID                 # local: <artifact_repo>/e129/<M>/<RUN_ID>/ -> data/e129-dl/<M>/<RUN_ID>/
"""
import argparse, json
from collections import Counter
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "configs/e129.yaml").read_text())


def rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def check_repo():
    from huggingface_hub import HfApi
    if not HfApi().repo_info(CFG["artifact_repo"], repo_type="model").private:
        raise SystemExit("artifact repo must be private")
    print("artifact repo private:", CFG["artifact_repo"])


def upload(data, run, model):
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=CFG["artifact_repo"], folder_path=data, path_in_repo=f"e129/{model}/{run}", allow_patterns=["*.jsonl", "states/*"])
    print("uploaded", data, "->", f"{CFG['artifact_repo']}/e129/{model}/{run}")


def fetch(model, run):
    import os
    from huggingface_hub import snapshot_download
    out = HERE / "data/e129-dl"
    snapshot_download(CFG["artifact_repo"], repo_type="model", allow_patterns=[f"e129/{model}/{run}/*"], local_dir=out, token=os.environ.get("HF_TOKEN"))
    print("fetched", out / "e129" / model / run)


def smoke_check(data, split):
    P = rows(Path(data) / f"prefix-{split}.jsonl"); C = {a: rows(Path(data) / f"cont-{split}-{a}.jsonl") for a in "UPT"}
    print(len(P), "prefixes; checkpoints", sum(bool(p["checkpoint"]) for p in P), "errors", [p["error"] for p in P if p["error"]])
    for a, R in C.items():
        print(a, len(R), "rows; errors", [r.get("error") for r in R if r.get("error")], "not applicable", sum(bool(r.get("not_applicable")) for r in R),
              "k_ids", Counter((r.get("k_ids") or {}).get("source") for r in R))
    assert P and not any(p["error"] for p in P) and any(p["checkpoint"] for p in P), "smoke prefixes"
    assert C["U"] and not any(r.get("error") for a in "UPT" for r in C[a]), "smoke continuations"


def summary(data, split, out):
    D = Path(data); P = rows(D / f"prefix-{split}.jsonl")
    s = dict(split=split, inputs=len(P), checkpoints=sum(bool(p["checkpoint"]) for p in P), prefix_errors=sum(bool(p["error"]) for p in P),
             checkpoint_call=Counter(p["checkpoint"]["k"] for p in P if p["checkpoint"]))
    for a in "UPT":
        R = rows(D / f"cont-{split}-{a}.jsonl")
        if not R:
            continue
        live = [r for r in R if not r.get("not_applicable")]
        s[a] = dict(rows=len(R), errors=sum(bool(r.get("error")) for r in live), not_applicable=len(R) - len(live),
                    episodes_with_parse_failure=sum((r.get("parse_failures") or 0) > 0 for r in live) / max(1, len(live)),
                    k_ids_source=Counter((r.get("k_ids") or {}).get("source") for r in live) if a == "U" else None)
        if split == "dev":   # dev outcomes may be read: they train the heads
            att = [r for r in live if r["attacked"] and r.get("security") is not None]; cl = [r for r in live if not r["attacked"] and r.get("utility") is not None]
            s[a].update(attack_success=sum(r["security"] for r in att) / max(1, len(att)), clean_completion=sum(r["utility"] for r in cl) / max(1, len(cl)),
                        attack_success_by_template={t: sum(r["security"] for r in att if r["attack"] == t) / max(1, sum(r["attack"] == t for r in att))
                                                    for t in sorted({r["attack"] for r in att})})
    if split == "dev":
        ck = [p["checkpoint"] for p in P if p["checkpoint"] and p["injection_task"]]
        s["attacked_checkpoint_exposed_step"] = sum(c["exposed_step"] for c in ck) / max(1, len(ck))
    Path(out).write_text(json.dumps(s, indent=1, default=str)); print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check-repo", "upload", "smoke-check", "summary", "fetch"])
    ap.add_argument("--data"); ap.add_argument("--run"); ap.add_argument("--out"); ap.add_argument("--model"); ap.add_argument("--split")
    a = ap.parse_args()
    {"check-repo": lambda: check_repo(), "upload": lambda: upload(a.data, a.run, a.model), "smoke-check": lambda: smoke_check(a.data, a.split),
     "summary": lambda: summary(a.data, a.split, a.out), "fetch": lambda: fetch(a.model, a.run)}[a.cmd]()
