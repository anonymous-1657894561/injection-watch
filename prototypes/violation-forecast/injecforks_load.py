"""The one InjecForks loader (decisions/, 2026-10-24): joins a config/split's rollouts with a label sidecar and the cuts,
and optionally restricts them to the shared evaluation subset, asserting every join.

    from injecforks_load import load
    rows = load("realistic", "dev")                     # labels v1.2, eval-v1.2 rows only (dev/test), from the Hub at tag v1.2
    rows = load("realistic", "train", subset=None)      # every row
    rows = load("smoke", "dev", root="data/e78/injecforks_v1")   # a local copy instead of the Hub
    rows = load("realistic-lookalike", "dev")           # E102 look-alikes (dev, test), from the Hub at tag v1.4

Without root=, it reads $INJECFORKS_ROOT if set, else data/injecforks/ at the repo root (the release archive, which it
unpacks there on first use if injecforks.tar.xz sits at the repo root), else the Hub.

Each row is the rollout row with the sidecar's columns on top (`violates` is the sidecar's label, `violates_v1` the
rollout file's original column) plus `cut` (N -> character offset, None if shorter) and `row_id` = config|split|case|rollout.
"""
import json
import os
from pathlib import Path

REPO, REVISION = "anonymous/InjecForks", "v1.2"
LOOKALIKE_REVISION = "v1.4"   # the look-alike configs exist from v1.4; every other file is unchanged since v1.2
K = {"train": 8, "dev": 8, "test": 16}


def _root(root, revision=REVISION):
    root = root or os.environ.get("INJECFORKS_ROOT")   # a local copy (e.g. the release archive) instead of the Hub
    if root:
        return Path(root)
    top = Path(__file__).resolve().parents[2]
    unpacked, archive = top / "data/injecforks", top / "injecforks.tar.xz"   # the release archive and where it unpacks
    if not (unpacked / "SHA256SUMS").is_file() and archive.is_file():
        import tarfile
        print(f"unpacking {archive.name} into {unpacked.parent} (first use)")
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked.parent)
    if (unpacked / "SHA256SUMS").is_file():
        return unpacked
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(REPO, repo_type="dataset", revision=revision))


VARIANT_REVISION = {"dev": "v1.5", "test": "v1.6"}   # E103's variant configs: dev T2/T4/T5 (v1.5), test T2/T4 (v1.6); rollouts, prompts, cuts; no labels


def load_variant(config, split="dev", root=None):
    """Every row of an E103 variant config, labelled by its rollout's own columns (violates = violates_broad, the realistic arm's
    label). Not for any frozen pass."""
    assert config.startswith("realistic-e103T") and split in VARIANT_REVISION, (config, split)
    R = _root(root, VARIANT_REVISION[split])
    rows = [json.loads(l) for l in open(R / config / f"{split}.jsonl")]
    cut = {(x["case"], x["rollout"]): x for x in map(json.loads, open(R / "cuts" / config / f"{split}.jsonl"))}
    assert len(rows) == len(cut) and len(rows) % K[split] == 0, ("row counts disagree", config, split)
    out = []
    for r in rows:
        c = cut[(r["case"], r["rollout"])]
        out.append({**r, "violates": r["violates_broad"], "cut": {int(n): o for n, o in c["cut"].items()},
                    "reasoning_tokens": c["reasoning_tokens"], "row_id": f"{config}|{split}|{r['case']}|{r['rollout']}"})
    return out


def load(config, split, labels="v1.2", subset="eval", root=None):
    assert subset in ("eval", "evalextra", None) and labels in ("v1.1", "v1.2")
    R = _root(root, LOOKALIKE_REVISION if config.endswith("-lookalike") else REVISION)
    if subset in ("eval", "evalextra"):
        assert split != "train", "the evaluation subset covers dev and test only"
    rows = [json.loads(l) for l in open(R / config / f"{split}.jsonl")]
    lab = {(x["case"], x["rollout"]): x for x in map(json.loads, open(R / f"labels-{labels}" / config / f"{split}.jsonl"))}
    cut = {(x["case"], x["rollout"]): x for x in map(json.loads, open(R / "cuts" / config / f"{split}.jsonl"))}
    assert len(rows) == len(lab) == len(cut) and len(rows) % K[split] == 0, ("row counts disagree", config, split)
    out = {}
    for r in rows:
        k = (r["case"], r["rollout"])
        l, c = lab[k], cut[k]  # KeyError = join failure
        x = {**r, "violates_v1": r["violates"], **{f: v for f, v in l.items() if f not in ("case", "rollout")},
             "cut": {int(n): o for n, o in c["cut"].items()}, "reasoning_tokens": c["reasoning_tokens"],
             "row_id": f"{config}|{split}|{r['case']}|{r['rollout']}"}
        out[k] = x
    assert len(out) == len(rows), "duplicate (case, rollout)"
    if subset is None:
        return list(out.values())
    ev = [json.loads(l) for l in open(R / ("eval-v1.2" if labels == "v1.2" else "eval") / config / f"{split}.jsonl")]
    sel = [out[(e["case"], e["rollout"])] for e in ev]  # KeyError = eval row missing from the split
    assert len({x["row_id"] for x in sel}) == len(ev), "duplicate eval rows"
    if labels == "v1.2" and not config.endswith(("-benign", "-lookalike")):  # the v1 eval/ was drawn on the v1 label
        by = {}
        for x in sel:
            by.setdefault(x["case"], []).append(x["violates"])
        assert all(any(v) and not all(v) for v in by.values()), "an eval case is not mixed under this label"
    if subset == "evalextra":   # the eval cases' other rollouts (test: 8 more per case), for scoring every rollout of those cases
        ids, cases = {x["row_id"] for x in sel}, {x["case"] for x in sel}
        extra = [x for x in out.values() if x["case"] in cases and x["row_id"] not in ids]
        assert len(extra) + len(sel) == len(cases) * K[split], ("eval cases do not have K rollouts", config, split)
        return extra
    return sel
