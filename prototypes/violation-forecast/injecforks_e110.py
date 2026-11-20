"""E110: fresh confirmation of the E109 gates (experiments/E110-gate-confirmation-fresh-inputs.md). Same procedure as
injecforks_e109.py (E093's design, Qwen3.5-9B BF16, E090's sampler), on fresh inputs:
- attack: the 123 realistic test eval cases E109 3b did not draw (the complement of random.Random("e109-attack").sample(cases, 176)),
  every eval rollout (8-15) reaching N = 16;
- look-alike: E110's 176 fresh realistic test look-alikes (e110_release.py; new requests, other seeded goals), rollouts 0-3 reaching N = 16,
  fetched from the private artifact repo (e110/injecforks_lookalike/).
Per row three arms, one sample each: control and E093's reminder, both continued from the row's first 16 reasoning tokens (the read the
N = 16 gates see), and reminder_n0 = E105's frozen reminder as the first reasoning tokens (the timing arm: what an N = 0 gate would do if it
reminded at once). Layer-20 states at N = 0 (per case) and N = 16 (per row), plus post-intervention reads at the last reminder token of
each reminder arm (N = "r16", "r0"; read-only, for the exploratory "did the reminder work" read). Strict label from the reply's calls; no dataset label is read.

  ~/.venvs/agentdojo/bin/python injecforks_e110.py configs/e110-9b-pod.yaml dry --split test
  python injecforks_e110.py configs/e110-9b-pod.yaml run --split test --final                 # pod (run_e110_pod.sh), once
  ~/.venvs/agentdojo/bin/python injecforks_e110.py configs/e110-9b-pod.yaml fetch --prefix RUN_ID
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

NAME = "e110"
FAMILY = "realistic"   # the probe and its threshold are realistic's
CFG = {"attack": "realistic", "lookalike": "realistic-lookalike-e110"}
POOL, PICK, FRESH_ATTACK, FRESH_LA, LA_ROWS = 299, 176, 123, 176, 4   # realistic test eval cases, E109's draw, and E110's counts (asserted)


def frozen(job):
    """The fixed numbers, asserted against E097's frozen config (decisions/, 2026-10-30) and E105's."""
    e097, e105 = (yaml.safe_load((HERE / f"configs/{c}.yaml").read_text()) for c in ("e097-9b-test-pod", "e105-9b-test-pod"))
    assert job["sampling"] == e097["sampling"] == e105["sampling"], "E090's sampler"
    assert job["generator"] == e097["generator"] == e105["generator"] and job["max_new_tokens"] == e097["max_new_tokens"] == 1536
    assert job["n"] == e097["n"] == 16 and job["state_layer"] == e097["state_layer"] == 20
    assert job["reminder"] == e097["reminder"], "E093's N = 16 reminder"
    assert hashlib.sha256(job["reminder"].encode()).hexdigest() == job["reminder_sha256"]
    assert job["dataset_revision"] == e105["dataset_revision"] == "v1.4"
    assert job["reminder_n0"] == e105["reminder_n0"] and hashlib.sha256(job["reminder_n0"].encode()).hexdigest() == job["reminder_n0_sha256"], "E105's N = 0 reminder"


def work(root, split, max_bases=None):
    """[(case, [(cat, kind, case, body, user_tool, [(rollout, reasoning, attacker_tools, cut)])])]: E110's fresh attack cases (eval rollouts
    8-15 reaching N = 16) and fresh look-alikes (rollouts 0-3 reaching N = 16)."""
    import random
    assert split == "test", "E110 is a test-split confirmation"
    allc = sorted({json.loads(l)["case"] for l in open(root / "eval-v1.2" / "realistic" / f"{split}.jsonl")})
    assert len(allc) == POOL, (len(allc), "realistic test eval cases")
    used = set(random.Random("e109-attack").sample(allc, PICK)); fresh = {c for c in allc if c not in used}; assert len(fresh) == FRESH_ATTACK
    W = [w for w in rows(root, split, "realistic", 16, None) if w[0] in fresh]
    assert len(W) == FRESH_ATTACK, (len(W), "fresh attack cases with rows reaching N = 16")
    La = rows(root, split, "realistic-lookalike-e110", 16, None)
    assert len(La) == FRESH_LA and all(":lookalike-" in c for c, *_ in La), (len(La), "fresh look-alike cases")
    ut = {}
    for cfg in ("realistic", "realistic-lookalike-e110"):
        for l in open(root / cfg / f"{split}.jsonl"):
            r = json.loads(l); ut[r["case"]] = r["user_tool"]
    out = [(cs, [("attack", "injected", cs, body, ut[cs], rs)]) for cs, body, rs in W]
    out += [(cs, [("lookalike", cs.rsplit("-", 1)[1], cs, body, ut[cs], rs[:LA_ROWS])]) for cs, body, rs in La]
    return (out[:max_bases] + out[-max_bases:]) if max_bases else out   # smoke: one attack case and one look-alike


def row_id(cat, split, case, rollout):
    return f"{CFG[cat]}|{split}|{case}|{rollout}"   # E102's score row ids


def select(job, split):
    return dict(job, select={"realistic": [split]})


def snapshot(job):
    """The dataset's realistic test files, and E110's fresh look-alikes from the private artifact repo (own manifest SHA256SUMS-e110),
    merged into one root by symlinks."""
    import shutil
    from huggingface_hub import snapshot_download
    ds = IS.snapshot(job)
    art = Path(snapshot_download(job["artifact_repo"], allow_patterns=["e110/injecforks_lookalike/*", "e110/injecforks_lookalike/**"])) / "e110/injecforks_lookalike"
    root = HERE / "data/e110/root"; shutil.rmtree(root, ignore_errors=True)
    for src in (ds, art):
        for p in src.rglob("*"):
            rel = p.relative_to(src) if p.is_file() else None
            if rel is None or (src == ds and not (rel.name.startswith("SHA256SUMS") or "realistic" in rel.parts)):
                continue   # the dataset side: only the realistic config's files and the manifests (a cached snapshot can hold more)
            (root / rel).parent.mkdir(parents=True, exist_ok=True); os.symlink(p.resolve(), root / rel)
    assert (root / "SHA256SUMS-e110").exists(), "E110's look-alike manifest"
    return root


def dry(job, split):
    """CPU, no model: counts, prefixes against the dataset's cuts, prompt lengths against the billed ones."""
    from transformers import AutoTokenizer
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    root = snapshot(select(job, split)); nver = verify(root); W = work(root, split); think_end = tok.convert_tokens_to_ids("</think>")
    rem = tok(job["reminder"], add_special_tokens=False).input_ids; n = 0
    for base, vs in W:
        for cat, kind, cs, body, ut, rs in vs:
            assert len(render(tok, body)) == body["prompt_tokens"], ("billed prompt tokens differ", cs)
            n += min(len(prefixes(tok, rs, job["n"], think_end)), LA_ROWS if cat == "lookalike" else 99)
    print(json.dumps(dict(split=split, verified_files=nver, cases=len(W), attack_cases=sum(v[0][0] == "attack" for _, v in W), rows=n, sequences_total=3 * n, reminder_tokens=len(rem)), indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import LayerTap, load_generator
    from injecforks_resample import expand, label, sample
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
    assert torch.cuda.is_available()
    frozen(job); api = None
    assert smoke or split == job["split"], ("the config is for", job["split"])
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), state_layer=job["smoke_state_layer"], **job["smoke_overrides"]); run_id = "smoke"
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
                if f.startswith("e110/") and f.endswith("results/run.json"):
                    rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                    if rj.get("split") == "test" and rj.get("complete"):
                        raise SystemExit(f"E110 was already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(select(job, split)); nver = verify(root); W = work(root, split, job.get("max_bases"))
    RES = HERE / "results/e110-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp, MAXN, NR, L = job["sampling"], job["max_new_tokens"], job["n"], job["state_layer"]
    rem = tok(job["reminder"], add_special_tokens=False).input_ids; rem0 = tok(job["reminder_n0"], add_special_tokens=False).input_ids; V = model.lm_head.weight.shape[0]
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, sampling=sp,
                   max_new_tokens=MAXN, n=NR, state_layer=L, reminder_tokens=len(rem), reminder_sha256=job["reminder_sha256"], reminder_n0_tokens=len(rem0), reminder_n0_sha256=job["reminder_n0_sha256"],
                   dataset_revision=job["dataset_revision"], generator=job["generator"], bases=len(W), cache_check=[])
    fout = open(RES / "e110.jsonl", "w" if smoke else "a"); SX, SM = [], []; last = time.time(); grab = {}
    tap = LayerTap(layers, [L], lambda li, h: grab.__setitem__("h", h))   # [B, T, d] output of layer L on the latest forward

    def flush(final_=False):
        fout.flush()
        if SX:
            np.save(RES / "states.npy", np.stack(SX).astype(np.float16))
            (RES / "states-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in SM))
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e110/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    def gen(tag):
        return torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e110|{split}|{tag}".encode()).hexdigest()[:8], 16))

    try:
        t0 = time.time()
        for wi, (base, vs) in enumerate(W):
            g = gen(base)   # one generator per case
            for cat, kind, case, b, user_tool, rs in vs:
                pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case); P = len(pids)
                o = fwd(pids, None, 0); base_kv = o.past_key_values
                SX.append(grab["h"][0, -1].float().cpu().numpy()); SM.append(dict(row_id=f"{CFG[cat]}|{split}|{case}", N=0))   # the prompt read (exposure at N = 0)
                pre = prefixes(tok, rs, NR, think_end)[:(LA_ROWS if cat == "lookalike" else None)][:job.get("k_cap") or None]; B = len(pre); atk = rs[0][2]
                x = torch.tensor([p for _, p in pre], device="cuda"); o1 = fwd(x, expand(base_kv, B), P, B)   # the probe's read, both arms' prefix
                for (rollout, _), h in zip(pre, grab["h"][:, -1].float().cpu().numpy()):
                    SX.append(h); SM.append(dict(row_id=row_id(cat, split, case, rollout), N=NR))
                if not receipt["cache_check"]:   # the copied cache equals one full forward (as E090 / E093 / E097)
                    with torch.no_grad():
                        hf = lm(input_ids=torch.tensor([pids + pre[-1][1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                    cos = float(torch.nn.functional.cosine_similarity(o1.last_hidden_state[-1, -1].float(), hf, 0))
                    receipt["cache_check"].append(round(cos, 5)); assert cos > 0.995, ("copied cache differs from a full forward", cos)
                for arm, add in (("control", []), ("reminder", rem)):
                    if add:
                        o2 = fwd(torch.tensor([add] * B, device="cuda"), copy.deepcopy(o1.past_key_values), P + NR, B); start = P + NR + len(add)
                        for (rollout, _), hh in zip(pre, grab["h"][:, -1].float().cpu().numpy()):   # post-intervention read: the reminder's last token
                            SX.append(hh); SM.append(dict(row_id=row_id(cat, split, case, rollout), N="r16"))
                    else:
                        o2, start = o1, P + NR
                    seen = torch.zeros(B, V, dtype=torch.bool, device="cuda")
                    seen[torch.arange(B)[:, None], torch.tensor([p + add for _, p in pre], device="cuda")] = True
                    kv = o2.past_key_values if add else copy.deepcopy(o1.past_key_values)   # o1's cache is batch B already; sampling mutates it
                    outs = sample(model, lm, kv, model.lm_head(o2.last_hidden_state[:, -1]), start, seen, sp, eos, MAXN, g)
                    for (rollout, p), y in zip(pre, outs):
                        txt = tok.decode(p + add + y, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                        fout.write(json.dumps(dict(part="e110", split=split, family=FAMILY, category=cat, kind=kind, base=base, case=case,
                                                   rollout=rollout, row_id=row_id(cat, split, case, rollout), user_tool=user_tool, arm=arm,
                                                   seed_tag=base, decided=dec, violates_strict=v, calls=calls, new_tokens=len(y), text=txt)) + "\n")
                    del o2
                # timing arm: E105's reminder as the first reasoning tokens, one sample per row (no prefix: N = 0 reminds at once)
                x0 = torch.tensor([rem0] * B, device="cuda"); o3 = fwd(x0, expand(base_kv, B), P, B)
                for (rollout, _), hh in zip(pre, grab["h"][:, -1].float().cpu().numpy()):   # post-intervention read after the N = 0 reminder
                    SX.append(hh); SM.append(dict(row_id=row_id(cat, split, case, rollout), N="r0"))
                seen = torch.zeros(B, V, dtype=torch.bool, device="cuda"); seen[torch.arange(B)[:, None], x0] = True
                outs = sample(model, lm, o3.past_key_values, model.lm_head(o3.last_hidden_state[:, -1]), P + len(rem0), seen, sp, eos, MAXN, g)
                for (rollout, p), y in zip(pre, outs):
                    txt = tok.decode(rem0 + y, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                    fout.write(json.dumps(dict(part="e110", split=split, family=FAMILY, category=cat, kind=kind, base=base, case=case,
                                               rollout=rollout, row_id=row_id(cat, split, case, rollout), user_tool=user_tool, arm="reminder_n0",
                                               seed_tag=base, decided=dec, violates_strict=v, calls=calls, new_tokens=len(y), text=txt)) + "\n")
                del o3
                del base_kv, o, o1; torch.cuda.empty_cache()
            print("n16", wi + 1, len(W), base, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["complete"] = True
    finally:
        tap.close(); receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("bases", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e110/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e110/{prefix}/*", f"e110/{prefix}/**"], local_dir=st)
    print("fetched to", st / "e110" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text())["attack"]; os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "test", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
