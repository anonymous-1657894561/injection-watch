"""E122: attack-style transfer of E115's frozen gates (experiments/E122-attack-style-transfer.md). States only, no generation: Qwen3.5-9B BF16
reads every E122 input (e122_release.py, private artifact repo e122/injecforks_e122/: 9 styles x 96 injected inputs + 24 look-alikes) and
the first 16 reasoning tokens of each of its rollouts reaching N = 16. States: N = 0 per case at every stored layer (8 12 16 20 24 26 28 30 32),
N = 16 per run at layer 20, as E115 stored them. The untreated outcome is the rollout's own (OpenRouter DeepInfra, E122 seeds); no reminder
arm. One pod, one one-time test read. No dataset label is read.

  ~/.venvs/agentdojo/bin/python injecforks_e122.py configs/e122-9b-pod.yaml dry --split test
  python injecforks_e122.py configs/e122-9b-pod.yaml run --split test --final                 # pod (run_e122_pod.sh)
  ~/.venvs/agentdojo/bin/python injecforks_e122.py configs/e122-9b-pod.yaml fetch --prefix RUN_ID
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_e097 import prefixes, rows
from injecforks_states import render, sha, verify
import testread

NAME = "e122"
STYLES = ["important", "tool_knowledge", "ignore_previous", "system_message", "injecagent_base", "injecagent_enhanced", "natural", "casual", "softened"]
CFG = {(s, la): f"realistic{'-lookalike' if la else ''}-e122-{s}" for s in STYLES for la in (False, True)}
LAYERS = [8, 12, 16, 20, 24, 26, 28, 30, 32]


def frozen(job):
    """The fixed numbers, asserted against E115's frozen config."""
    e115 = yaml.safe_load((HERE / "configs/e115-9b-pod-s0.yaml").read_text())["attack"]
    for k in ("generator", "n", "state_layer", "n0_layers"):
        assert job[k] == e115[k], k
    assert job["n0_layers"] == LAYERS


def work(root, split, max_bases=None):
    """[(cfg, case, body, [(rollout, reasoning, attacker_tools, cut)])]: every rollout reaching N = 16"""
    assert split == "test", "E122 is a test-split read"
    out = []
    for (s, la), cfg in CFG.items():
        W = rows(root, split, cfg, 16, None)
        assert len({json.loads(l)["case"] for l in open(root / cfg / f"{split}.jsonl")}) == (24 if la else 96), cfg
        out += [(cfg, cs, body, rs) for cs, body, rs in W]
    return out[:max_bases] if max_bases else out


def snapshot(job):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(job["artifact_repo"], allow_patterns=["e122/injecforks_e122/*", "e122/injecforks_e122/**"])) / "e122/injecforks_e122"
    assert (root / "SHA256SUMS-e122").exists(), "E122's manifest"
    return root


def dry(job, split):
    from transformers import AutoTokenizer
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(job); nver = verify(root); think_end = tok.convert_tokens_to_ids("</think>"); n = toks = 0
    W = work(root, split)
    for cfg, cs, body, rs in W:
        assert len(render(tok, body)) == body["prompt_tokens"], ("billed prompt tokens differ", cs)
        n += len(prefixes(tok, rs, job["n"], think_end)); toks += body["prompt_tokens"]
    print(json.dumps(dict(split=split, verified_files=nver, inputs=len(W), runs16=n, prompt_tokens=toks), indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import LayerTap, load_generator
    from injecforks_resample import expand
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
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
        for f in api.list_repo_files(job["artifact_repo"]):   # the pod has no local receipt: a completed run is the receipt
            if f.startswith("e122/") and f.endswith("results/run.json"):
                rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                if rj.get("split") == "test" and rj.get("complete"):
                    raise SystemExit(f"E122 was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(job); nver = verify(root); W = work(root, split, job.get("max_bases"))
    RES = HERE / "results/e122-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    think_end = tok.convert_tokens_to_ids("</think>"); NR, L, L0 = job["n"], job["state_layer"], job["n0_layers"]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, n=NR, state_layer=L,
                   n0_layers=L0, generator=job["generator"], inputs=len(W), cache_check=[])
    S0, M0, S16, M16 = [], [], [], []; last = time.time(); grab = {}
    tap = LayerTap(layers, sorted(set(L0) | {L}), lambda li, h: grab.__setitem__(li, h))

    def flush(final_=False):
        if S0:
            np.save(RES / "states0.npy", np.stack(S0).astype(np.float16)); (RES / "states0-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M0))
        if S16:
            np.save(RES / "states.npy", np.stack(S16).astype(np.float16)); (RES / "states-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M16))
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e122/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    try:
        t0 = time.time()
        for wi, (cfg, case, b, rs) in enumerate(W):
            pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
            o = fwd(pids, None, 0); base_kv = o.past_key_values
            S0.append(np.stack([grab[li][0, -1].float().cpu().numpy() for li in L0])); M0.append(dict(case=f"{cfg}|{split}|{case}", N=0, layers=L0))
            pre = prefixes(tok, rs, NR, think_end)[:job.get("k_cap") or None]; B = len(pre)
            if B:
                x = torch.tensor([p for _, p in pre], device="cuda"); o1 = fwd(x, expand(base_kv, B), P, B)
                for (rollout, _), h in zip(pre, grab[L][:, -1].float().cpu().numpy()):
                    S16.append(h); M16.append(dict(row_id=f"{cfg}|{split}|{case}|{rollout}", N=NR))
                if not receipt["cache_check"]:
                    with torch.no_grad():
                        hf = lm(input_ids=torch.tensor([pids + pre[-1][1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                    cos = float(torch.nn.functional.cosine_similarity(o1.last_hidden_state[-1, -1].float(), hf, 0))
                    receipt["cache_check"].append(round(cos, 5)); assert cos > 0.995, ("copied cache differs from a full forward", cos)
                del o1
            del o, base_kv; torch.cuda.empty_cache()
            if (wi + 1) % 20 == 0:
                print("e122", wi + 1, len(W), cfg, case, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["complete"] = True
    finally:
        tap.close(); receipt["finished"] = time.time(); flush(final_=True)
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("inputs", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e122/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e122/{prefix}/*", f"e122/{prefix}/**"], local_dir=st)
    print("fetched to", st / "e122" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text())["attack"]; os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "test", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
