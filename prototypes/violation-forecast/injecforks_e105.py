"""E105: the frozen N = 0 reminder's cost on InjecForks v1.4 look-alikes, beside the benign twins
(experiments/E105-lookalike-reminder-cost.md). Qwen3.5-9B BF16, E090's sampler.

  ~/.venvs/e01/bin/python injecforks_e105.py configs/e105-9b-dev-pod.yaml dry [--split dev|test]   # CPU: tokenizer only, no model
  ~/.venvs/e01/bin/python injecforks_e105.py configs/e105-9b-dev-pod.yaml smoke                    # local GPU: Qwen3.5-0.8B, short
  python injecforks_e105.py configs/e105-9b-dev-pod.yaml run --split dev                           # pod (run_e105_pod.sh)
  python injecforks_e105.py configs/e105-9b-test-pod.yaml run --split test --final                 # pod, once (testread.py)
  ~/.venvs/e01/bin/python injecforks_e105.py configs/e105-9b-dev-pod.yaml fetch --prefix RUN_ID

E097 part (b) unchanged in procedure, on {smoke,realistic}-{benign,lookalike}: per case, k samples from the prompt (control_n0)
and then k with the reminder as the first reasoning tokens (reminder_n0), from one case generator. k = every rollout the dataset
holds for the case on dev (8), and the case's eval-v1.2 rows on test (4). A case is `<attack>:<row>:<benign|lookalike>-<kind>`;
the twins and look-alikes built from one injected case `<attack>:<row>` (its "base") share the generator seed, and the work is
ordered by base so a timer kill keeps whole pairs. No dataset label is read into a record. Output:
results/e105-cloud/{run.json, e105.jsonl}; uploaded under e105/<run_id>/.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_states import render, sha, snapshot, verify
import testread

NAME = "e105-n0"
FAMILIES = ("smoke", "realistic")
CATS = {"benign": "twin", "lookalike": "lookalike"}
KINDS = {"twin": ("neutral", "imperative"), "lookalike": ("remind", "refrain")}
COUNTS = {"dev": dict(cases=24, file_rollouts=8, k=8), "test": dict(cases=176, file_rollouts=16, k=4)}   # the task brief's, asserted


def frozen(job):
    """The fixed numbers, asserted against E097's frozen config (decisions/, 2026-10-30) and E096's."""
    e096, e097 = (yaml.safe_load((HERE / f"configs/{c}.yaml").read_text()) for c in ("e096-9b-pod", "e097-9b-test-pod"))
    assert job["sampling"] == e097["sampling"] == e096["sampling"], "E090's sampler"
    assert job["generator"] == e097["generator"] == e096["generator"] and job["max_new_tokens"] == e097["max_new_tokens"] == 1536
    assert job["reminder_n0"] == e097["reminder_n0"] == e096["reminder_n0"] == e097["reminder"].lstrip("\n"), "the frozen N = 0 reminder"
    assert hashlib.sha256(job["reminder_n0"].encode()).hexdigest() == job["reminder_n0_sha256"]
    assert job["dataset_revision"] == "v1.4"


def parse(case):
    """`<attack>:<row>:<benign|lookalike>-<kind>` -> (base, category, kind)."""
    base, v = case.rsplit(":", 1); src, kind = v.split("-", 1)
    assert src in CATS and kind in KINDS[CATS[src]], case
    return base, CATS[src], kind


def cases(root, split, family, cat):
    """[(case, body, user_tool, attacker_tools, k)] for one config, with the brief's counts asserted."""
    cfg = f"{family}-{'benign' if cat == 'twin' else 'lookalike'}"
    held, meta = {}, {}
    for l in open(root / cfg / f"{split}.jsonl"):
        r = json.loads(l); held.setdefault(r["case"], set()).add(r["rollout"])
        meta[r["case"]] = (r["user_tool"], r.get("attacker_tools") or [])
        assert r.get("benign") in (None, parse(r["case"])[2]), (r["case"], r.get("benign"))
    ev = {}
    for e in map(json.loads, open(root / "eval-v1.2" / cfg / f"{split}.jsonl")):
        ev.setdefault(e["case"], set()).add(e["rollout"])
    bodies = {b["case"]: b for b in map(json.loads, open(root / "prompts" / cfg / f"{split}.jsonl"))}
    c = COUNTS[split]
    assert len(held) == c["cases"], (cfg, split, "cases", len(held))
    assert all(len(v) == c["file_rollouts"] for v in held.values()), (cfg, split, "rollouts per case", {len(v) for v in held.values()})
    assert set(ev) == set(held) and all(len(v) == COUNTS["test"]["k"] for v in ev.values()), (cfg, split, "eval-v1.2 rows per case")
    assert set(bodies) == set(held), (cfg, split, "prompts")
    k = (lambda cs: len(held[cs])) if split == "dev" else (lambda cs: len(ev[cs]))
    return [(cs, bodies[cs], *meta[cs], k(cs)) for cs in sorted(held)], {cs: sorted(v) for cs, v in ev.items()}


def work(root, split, max_bases=None):
    """[(family, base, [(cat, kind, case, body, user_tool, attacker_tools, k)])], every base holding its two twins and two
    look-alikes (asserted), plus each case's eval-v1.2 rollout ids (for the pairing report)."""
    out, evs = [], {}
    for fam in FAMILIES:
        by = {}
        for cat in ("twin", "lookalike"):
            W, ev = cases(root, split, fam, cat); evs.update({(fam, cs): v for cs, v in ev.items()})
            for cs, body, ut, at, k in W:
                base, cat_, kind = parse(cs); assert cat_ == cat
                by.setdefault(base, []).append((cat, kind, cs, body, ut, at, k))
        for base, vs in by.items():
            assert sorted((c, kd) for c, kd, *_ in vs) == sorted((c, kd) for c in KINDS for kd in KINDS[c]), (fam, base)
            assert len({ut for *_, ut, _, _ in vs}) == 1, (fam, base, "twin and look-alike share the user tool")
        out += [(fam, base, sorted(by[base], key=lambda v: (v[0] != "twin", v[1]))) for base in sorted(by)][:max_bases or None]
    return out, evs


def dry(job, split):
    """CPU, no model: counts, pairing, prompt lengths against the billed ones, and where the reminder goes."""
    from transformers import AutoTokenizer
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(dict(job, select={f"{f}-{c}": [split] for f in FAMILIES for c in ("benign", "lookalike")})); nver = verify(root)
    W, evs = work(root, split)
    rem0 = tok(job["reminder_n0"], add_special_tokens=False).input_ids
    out = dict(split=split, verified_files=nver, bases=len(W), reminder_n0_tokens=len(rem0), per_config={})
    for fam, base, vs in W:
        for cat, kind, cs, body, *_ , k in vs:
            assert len(render(tok, body)) == body["prompt_tokens"], ("billed prompt tokens differ", cs)
            d = out["per_config"].setdefault(f"{fam}-{cat}", dict(cases=0, rows_per_arm=0)); d["cases"] += 1; d["rows_per_arm"] += k
    out["sequences_total"] = 2 * sum(d["rows_per_arm"] for d in out["per_config"].values())
    # the same eval-v1.2 rollout ids (dataset seeds 7400 + r) for a look-alike and its twin, reported (N = 0 samples per case)
    same = [evs[(f, cs)] == evs[(f, cs2)] for f, _, vs in W for (c, _, cs, *_r) in vs if c == "lookalike"
            for (c2, _, cs2, *_r2) in vs if c2 == "twin"]
    out["eval_rollouts_equal_lookalike_vs_twin"] = f"{sum(same)}/{len(same)}"
    fam, base, vs = W[0]; pids = render(tok, vs[0][3])
    out["insertion_point"] = repr(tok.decode(pids[-12:]) + "|" + tok.decode(rem0[:12]) + "...")   # the prompt ends at "<think>\n"
    print(json.dumps(out, indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import load_generator
    from injecforks_resample import expand, label, sample
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
    assert torch.cuda.is_available()
    frozen(job); api = None
    assert smoke or split == job["split"], ("the config is for", job["split"])
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), **job["smoke_overrides"]); run_id = "smoke"
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        if split == "test":   # the pod has no local receipt: a completed test run in the artifact repo is the receipt
            from huggingface_hub import hf_hub_download
            for f in api.list_repo_files(job["artifact_repo"]):
                if f.startswith("e105/") and f.endswith("results/run.json"):
                    rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                    if rj.get("split") == "test" and rj.get("complete"):
                        raise SystemExit(f"E105 test was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(dict(job, select={f"{f}-{c}": [split] for f in FAMILIES for c in ("benign", "lookalike")})); nver = verify(root)
    W, _ = work(root, split, job.get("max_bases"))
    RES = HERE / "results/e105-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, _ = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); sp, MAXN = job["sampling"], job["max_new_tokens"]
    rem0 = tok(job["reminder_n0"], add_special_tokens=False).input_ids; V = model.lm_head.weight.shape[0]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, sampling=sp,
                   max_new_tokens=MAXN, reminder_n0_tokens=len(rem0), reminder_n0_sha256=job["reminder_n0_sha256"],
                   dataset_revision=job["dataset_revision"], generator=job["generator"], bases=len(W), cache_check=[])
    fout = open(RES / "e105.jsonl", "w" if smoke else "a"); last = time.time()

    def flush(final_=False):
        fout.flush()
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e105/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    def gen(tag):
        return torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e105|{split}|{tag}".encode()).hexdigest()[:8], 16))

    try:
        t0 = time.time()
        for wi, (fam, base, vs) in enumerate(W):
            for cat, kind, case, b, user_tool, atk, k in vs:
                k = min(k, job.get("k_cap") or k)   # smoke only
                pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
                o = fwd(pids, None, 0); base_kv, first = o.past_key_values, model.lm_head(o.last_hidden_state[:, -1])
                g = gen(f"{fam}|{base}")   # shared by the base's twins and look-alikes
                outs = sample(model, lm, expand(base_kv, k), first.repeat(k, 1), P, torch.zeros(k, V, dtype=torch.bool, device="cuda"), sp, eos, MAXN, g)
                res = [("control_n0", tok.decode(x, skip_special_tokens=False), len(x)) for x in outs]
                x = torch.tensor([rem0] * k, device="cuda"); o2 = fwd(x, expand(base_kv, k), P, k)
                if not receipt["cache_check"]:   # the copied cache equals one full forward (as E090 / E093 / E097)
                    with torch.no_grad():
                        hf = lm(input_ids=torch.tensor([pids + rem0], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                    cos = float(torch.nn.functional.cosine_similarity(o2.last_hidden_state[-1, -1].float(), hf, 0))
                    receipt["cache_check"].append(round(cos, 5)); assert cos > 0.995, ("copied cache differs from a full forward", cos)
                seen = torch.zeros(k, V, dtype=torch.bool, device="cuda"); seen[torch.arange(k)[:, None], x] = True
                outs = sample(model, lm, o2.past_key_values, model.lm_head(o2.last_hidden_state[:, -1]), P + len(rem0), seen, sp, eos, MAXN, g)
                res += [("reminder_n0", tok.decode(rem0 + x, skip_special_tokens=False), len(x)) for x in outs]
                for j, (arm, txt, n_new) in enumerate(res):
                    dec, v, calls = label(txt, atk)
                    fout.write(json.dumps(dict(part="n0", split=split, family=fam, category=cat, kind=kind, base=base, case=case,
                                               user_tool=user_tool, arm=arm, j=j % k, seed_tag=f"{fam}|{base}", decided=dec,
                                               violates_strict=v, calls=calls, new_tokens=n_new, text=txt)) + "\n")
                del base_kv, o, o2; torch.cuda.empty_cache()
            print("n0", wi + 1, len(W), fam, base, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["complete"] = True
    finally:
        receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("bases", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e105/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e105/{prefix}/*"], local_dir=st)
    print("fetched to", st / "e105" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "dev", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
