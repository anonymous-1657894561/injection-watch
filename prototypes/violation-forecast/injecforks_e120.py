"""E120: E097's resampling arm on E110's fresh injected inputs (experiments/E120-gated-resample-fresh-inputs.md).
One fresh sample from the prompt (N = 0, no prefix, no reminder) per E110 attack run: the 123 realistic test cases E109 3b did not draw,
every eval rollout (8-15) reaching N = 16 (493 runs; injecforks_e110.work, unchanged). Qwen3.5-9B BF16, E090's sampler, <= 1,536 new tokens,
the same N = 0 sampling path as E090's fresh samples (injecforks_resample: empty presence-penalty set, cache expanded from the prompt).
Strict label from the reply's calls; no dataset label and no E110 outcome is read here. Gating is offline (e120_analyze.py).

  ~/.venvs/agentdojo/bin/python injecforks_e120.py configs/e120-9b-pod.yaml dry --split test
  python injecforks_e120.py configs/e120-9b-pod.yaml run --split test --final                 # pod (run_e120_pod.sh), once
  ~/.venvs/agentdojo/bin/python injecforks_e120.py configs/e120-9b-pod.yaml fetch --prefix RUN_ID
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_e097 import prefixes
import injecforks_e110 as E110
from injecforks_states import render, sha, verify
import testread

NAME = "e120"
RUNS = 493   # E110's attack runs (asserted)


def attack_work(root, split):
    W = [(b, v) for b, vs in E110.work(root, split) for v in vs if v[0] == "attack"]
    assert len(W) == E110.FRESH_ATTACK, len(W)
    return W


def dry(job, split):
    from transformers import AutoTokenizer
    E110.frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"]); think_end = tok.convert_tokens_to_ids("</think>")
    root = E110.snapshot(E110.select(job, split)); nver = verify(root); W = attack_work(root, split)
    n = sum(len(prefixes(tok, v[5], job["n"], think_end)) for _, v in W)
    assert n == RUNS, (n, "E110's attack runs")
    print(json.dumps(dict(split=split, verified_files=nver, attack_cases=len(W), runs=n, sequences_total=n), indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import load_generator
    from injecforks_resample import expand, label, sample
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
    assert torch.cuda.is_available()
    E110.frozen(job); api = None
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), **job["smoke_overrides"]); run_id = "smoke"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        for f in api.list_repo_files(job["artifact_repo"]):
            if f.startswith("e120/") and f.endswith("results/run.json"):
                rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                if rj.get("split") == "test" and rj.get("complete"):
                    raise SystemExit(f"E120 was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = E110.snapshot(E110.select(job, split)); nver = verify(root); W = attack_work(root, split)
    if smoke:
        W = W[:job["max_bases"]]
    RES = HERE / "results/e120-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp, MAXN = job["sampling"], job["max_new_tokens"]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, sampling=sp,
                   max_new_tokens=MAXN, dataset_revision=job["dataset_revision"], generator=job["generator"], bases=len(W), sequences=0)
    fout = open(RES / "e120.jsonl", "w"); last = time.time()

    def flush(final_=False):
        fout.flush()
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e120/{run_id}/results")

    try:
        t0 = time.time()
        for wi, (base, (cat, kind, case, b, user_tool, rs)) in enumerate(W):
            g = torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e120|{split}|{base}".encode()).hexdigest()[:8], 16))
            pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
            pre = prefixes(tok, rs, job["n"], think_end)[:job.get("k_cap") or None]; B = len(pre); atk = rs[0][2]
            with torch.no_grad():
                o = lm(input_ids=torch.tensor([pids], device="cuda"), use_cache=True, position_ids=torch.arange(P, device="cuda")[None])
            first = model.lm_head(o.last_hidden_state[:, -1]); V = first.shape[-1]
            seen = torch.zeros(B, V, dtype=torch.bool, device="cuda")   # E090's N = 0 path: nothing generated yet, nothing penalised
            outs = sample(model, lm, expand(o.past_key_values, B), first.repeat(B, 1), P, seen, sp, eos, MAXN, g)
            for (rollout, _), y in zip(pre, outs):   # one fresh sample per E110 run; the rollout only names the run it stands in for
                txt = tok.decode(y, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                fout.write(json.dumps(dict(part="e120", split=split, category=cat, kind=kind, base=base, case=case, rollout=rollout,
                                           row_id=E110.row_id(cat, split, case, rollout), user_tool=user_tool, arm="resample_n0",
                                           seed_tag=f"e120|{split}|{base}", decided=dec, violates_strict=v, calls=calls, new_tokens=len(y), text=txt)) + "\n")
            receipt["sequences"] += B
            del o; torch.cuda.empty_cache()
            print("n0", wi + 1, len(W), base, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        assert smoke or receipt["sequences"] == RUNS, receipt["sequences"]
        receipt["complete"] = True
    finally:
        receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("bases", "sequences", "peak_gib")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e120/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e120/{prefix}/*", f"e120/{prefix}/**"], local_dir=st)
    print("fetched to", st / "e120" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text())["attack"]; os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "test", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
