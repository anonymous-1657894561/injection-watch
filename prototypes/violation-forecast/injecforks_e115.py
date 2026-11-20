"""E115: confirmation of the prompt-only risk gate on six new carriers (experiments/E115-prompt-risk-gate-confirmation.md). Same procedure as
injecforks_e110.py (E093's design, Qwen3.5-9B BF16, E090's sampler, E110's three arms), on E115's fresh inputs (e115_release.py, private
artifact repo e115/injecforks_e115/): injected inputs, benign twins and look-alikes, every rollout (0-7) reaching N = 16.
Per run three arms, one sample each: control and E093's reminder continued from the run's first 16 reasoning tokens, and reminder_n0 = E105's
reminder as the first reasoning tokens. States: N = 0 per case at every stored layer (8 12 16 20 24 26 28 30 32: the frozen heads read 12,
20, 24), N = 16 per run at layer 20. Decode time per arm (latency). Sharded by source task (carrier x goal: an injected input with its twins
and look-alikes) into `shards` pods; each shard is its own one-time test read. Strict label from the reply's calls; no dataset label is read.

  ~/.venvs/agentdojo/bin/python injecforks_e115.py configs/e115-9b-pod-s0.yaml dry --split test
  python injecforks_e115.py configs/e115-9b-pod-s0.yaml run --split test --final                 # pod (run_e115_pod.sh), once per shard
  ~/.venvs/agentdojo/bin/python injecforks_e115.py configs/e115-9b-pod-s0.yaml fetch --prefix RUN_ID
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_e097 import prefixes, rows
import injecforks_states as IS
from injecforks_states import render, sha, verify
import testread

NAME = "e115"
FAMILY = "realistic"
CFG = {"attack": "realistic-e115", "twin": "realistic-benign-e115", "lookalike": "realistic-lookalike-e115"}
CASES = {"attack": 372, "twin": 96, "lookalike": 96}
LAYERS = [8, 12, 16, 20, 24, 26, 28, 30, 32]


def frozen(job):
    """The fixed numbers, asserted against E110's frozen config (itself asserted against E097's and E105's)."""
    e110 = yaml.safe_load((HERE / "configs/e110-9b-pod.yaml").read_text())["attack"]
    for k in ("sampling", "generator", "max_new_tokens", "n", "state_layer", "reminder", "reminder_sha256", "reminder_n0", "reminder_n0_sha256"):
        assert job[k] == e110[k], k
    assert hashlib.sha256(job["reminder"].encode()).hexdigest() == job["reminder_sha256"]
    assert hashlib.sha256(job["reminder_n0"].encode()).hexdigest() == job["reminder_n0_sha256"]
    assert job["n0_layers"] == LAYERS and 0 <= job["shard"] < job["shards"] == 4


def source_task(case):
    return case.split(":lookalike-")[0].split(":benign-")[0]


def shard_of(case, shards):
    return int(hashlib.sha256(f"e115|{source_task(case)}".encode()).hexdigest()[:8], 16) % shards


def work(root, split, shard, shards, max_bases=None):
    """[(cat, case, body, user_tool, [(rollout, reasoning, attacker_tools, cut)])] of this shard: every rollout reaching N = 16."""
    assert split == "test", "E115 is a test-split confirmation"
    out = []
    for cat, cfg in CFG.items():
        W = rows(root, split, cfg, 16, None)
        allc = {json.loads(l)["case"] for l in open(root / cfg / f"{split}.jsonl")}
        assert len(allc) == CASES[cat], (cat, len(allc))
        ut = {}
        for l in open(root / cfg / f"{split}.jsonl"):
            r = json.loads(l); ut[r["case"]] = r["user_tool"]
        out += [(cat, cs, body, ut[cs], rs) for cs, body, rs in W if shard_of(cs, shards) == shard]
    if max_bases:   # smoke: one input per category
        out = [next(w for w in out if w[0] == c) for c in CFG]
    return out


def row_id(cat, split, case, rollout):
    return f"{CFG[cat]}|{split}|{case}|{rollout}"


def snapshot(job):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(job["artifact_repo"], allow_patterns=["e115/injecforks_e115/*", "e115/injecforks_e115/**"])) / "e115/injecforks_e115"
    assert (root / "SHA256SUMS-e115").exists(), "E115's manifest"
    return root


def dry(job, split):
    """CPU, no model: counts, prefixes against the dataset's cuts, prompt lengths against the billed ones, per shard."""
    from transformers import AutoTokenizer
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(job); nver = verify(root); think_end = tok.convert_tokens_to_ids("</think>"); per = {}
    for s in range(job["shards"]):
        n = 0
        for cat, cs, body, ut, rs in work(root, split, s, job["shards"]):
            assert len(render(tok, body)) == body["prompt_tokens"], ("billed prompt tokens differ", cs)
            n += len(prefixes(tok, rs, job["n"], think_end))
        per[s] = dict(runs=n, sequences=3 * n)
    print(json.dumps(dict(split=split, verified_files=nver, shards=per, sequences_total=sum(v["sequences"] for v in per.values())), indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import LayerTap, load_generator
    from injecforks_resample import expand, label, sample
    shard = job["shard"]; name = f"{NAME}-s{shard}"
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), name)
    assert torch.cuda.is_available()
    frozen(job); api = None
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), state_layer=job["smoke_state_layer"], n0_layers=job["smoke_n0_layers"], **job["smoke_overrides"]); run_id = "smoke"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        for f in api.list_repo_files(job["artifact_repo"]):   # the pod has no local receipt: a completed run of this shard is the receipt
            if f.startswith("e115/") and f.endswith("results/run.json"):
                rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                if rj.get("split") == "test" and rj.get("complete") and rj.get("shard") == shard:
                    raise SystemExit(f"E115 shard {shard} was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(job); nver = verify(root); W = work(root, split, shard, job["shards"], job.get("max_bases"))
    RES = HERE / "results/e115-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp, MAXN, NR, L, L0 = job["sampling"], job["max_new_tokens"], job["n"], job["state_layer"], job["n0_layers"]
    rem = tok(job["reminder"], add_special_tokens=False).input_ids; rem0 = tok(job["reminder_n0"], add_special_tokens=False).input_ids; V = model.lm_head.weight.shape[0]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, shard=shard, shards=job["shards"], started=time.time(), verified_files=nver,
                   sampling=sp, max_new_tokens=MAXN, n=NR, state_layer=L, n0_layers=L0, reminder_sha256=job["reminder_sha256"], reminder_n0_sha256=job["reminder_n0_sha256"],
                   generator=job["generator"], inputs=len(W), cache_check=[], timing=[])
    fout = open(RES / "e115.jsonl", "w" if smoke else "a"); S0, M0, S16, M16 = [], [], [], []; last = time.time(); grab = {}
    tap = LayerTap(layers, sorted(set(L0) | {L}), lambda li, h: grab.__setitem__(li, h))

    def flush(final_=False):
        fout.flush()
        if S0:
            np.save(RES / "states0.npy", np.stack(S0).astype(np.float16)); (RES / "states0-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M0))
        if S16:
            np.save(RES / "states.npy", np.stack(S16).astype(np.float16)); (RES / "states-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M16))
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e115/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    def gen(tag):
        return torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e115|{split}|{tag}".encode()).hexdigest()[:8], 16))

    def timed(fn):
        torch.cuda.synchronize(); t = time.time(); out = fn(); torch.cuda.synchronize(); return out, time.time() - t

    def write(cat, kind, case, user_tool, rollout, arm, txt, atk, y, secs):
        dec, v, calls = label(txt, atk)
        fout.write(json.dumps(dict(part="e115", split=split, family=FAMILY, category=cat, kind=kind, source_task=source_task(case), case=case,
                                   rollout=rollout, row_id=row_id(cat, split, case, rollout), user_tool=user_tool, arm=arm, seed_tag=case, decided=dec,
                                   violates_strict=v, calls=calls, new_tokens=len(y), batch_decode_s=round(secs, 3), text=txt)) + "\n")

    try:
        t0 = time.time()
        for wi, (cat, case, b, user_tool, rs) in enumerate(W):
            g = gen(case); kind = case.rsplit("-", 1)[1] if cat != "attack" else "injected"
            pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
            o, t_pre = timed(lambda: fwd(pids, None, 0)); base_kv = o.past_key_values
            S0.append(np.stack([grab[li][0, -1].float().cpu().numpy() for li in L0])); M0.append(dict(case=f"{CFG[cat]}|{split}|{case}", N=0, layers=L0))
            pre = prefixes(tok, rs, NR, think_end)[:job.get("k_cap") or None]; B = len(pre); atk = rs[0][2]
            x = torch.tensor([p for _, p in pre], device="cuda"); o1 = fwd(x, expand(base_kv, B), P, B)
            for (rollout, _), h in zip(pre, grab[L][:, -1].float().cpu().numpy()):
                S16.append(h); M16.append(dict(row_id=row_id(cat, split, case, rollout), N=NR))
            if not receipt["cache_check"]:
                with torch.no_grad():
                    hf = lm(input_ids=torch.tensor([pids + pre[-1][1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                cos = float(torch.nn.functional.cosine_similarity(o1.last_hidden_state[-1, -1].float(), hf, 0))
                receipt["cache_check"].append(round(cos, 5)); assert cos > 0.995, ("copied cache differs from a full forward", cos)
            tm = dict(case=case, prompt_tokens=P, prefill_s=round(t_pre, 4), B=B)
            for arm, add in (("control", []), ("reminder", rem)):
                if add:
                    o2 = fwd(torch.tensor([add] * B, device="cuda"), copy.deepcopy(o1.past_key_values), P + NR, B); start = P + NR + len(add)
                else:
                    o2, start = o1, P + NR
                seen = torch.zeros(B, V, dtype=torch.bool, device="cuda")
                seen[torch.arange(B)[:, None], torch.tensor([p + add for _, p in pre], device="cuda")] = True
                kv = o2.past_key_values if add else copy.deepcopy(o1.past_key_values)
                outs, secs = timed(lambda: sample(model, lm, kv, model.lm_head(o2.last_hidden_state[:, -1]), start, seen, sp, eos, MAXN, g))
                tm[f"{arm}_s"] = round(secs, 3); tm[f"{arm}_steps"] = max(len(y) for y in outs)
                for (rollout, p), y in zip(pre, outs):
                    write(cat, kind, case, user_tool, rollout, arm, tok.decode(p + add + y, skip_special_tokens=False), atk, y, secs)
                del o2
            x0 = torch.tensor([rem0] * B, device="cuda"); o3 = fwd(x0, expand(base_kv, B), P, B)
            seen = torch.zeros(B, V, dtype=torch.bool, device="cuda"); seen[torch.arange(B)[:, None], x0] = True
            outs, secs = timed(lambda: sample(model, lm, o3.past_key_values, model.lm_head(o3.last_hidden_state[:, -1]), P + len(rem0), seen, sp, eos, MAXN, g))
            tm["reminder_n0_s"] = round(secs, 3); tm["reminder_n0_steps"] = max(len(y) for y in outs); receipt["timing"].append(tm)
            for (rollout, p), y in zip(pre, outs):
                write(cat, kind, case, user_tool, rollout, "reminder_n0", tok.decode(rem0 + y, skip_special_tokens=False), atk, y, secs)
            del o3, base_kv, o, o1; torch.cuda.empty_cache()
            print("e115", wi + 1, len(W), cat, case, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["complete"] = True
    finally:
        tap.close(); receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), name, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("inputs", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e115/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e115/{prefix}/*", f"e115/{prefix}/**"], local_dir=st)
    print("fetched to", st / "e115" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text())["attack"]; os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "test", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
