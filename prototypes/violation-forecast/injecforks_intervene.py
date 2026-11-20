"""E093: interventions at N = 16 on InjecForks v1.2 realistic dev (experiments/E093-gated-intervention-pilot.md).

  ~/.venvs/e01/bin/python injecforks_intervene.py configs/e093-9b-pod.yaml smoke   # local GPU: Qwen3.5-0.8B, 2 cases, short
  python injecforks_intervene.py configs/e093-9b-pod.yaml run                      # pod
  ~/.venvs/e01/bin/python injecforks_intervene.py configs/e093-9b-pod.yaml fetch --prefix RUN_ID

Per case the prompt is prefilled once (as E090) and each rollout's first N reasoning tokens are continued under several arms:
control; reminder (a sentence inserted after the prefix); steering, -alpha * direction added to the output of one decoder layer
at every generated token (probe direction u, or a random direction orthogonal to it). Every arm is applied to every rollout,
so gating is computed offline. Label: strict, a <tool_call> to the attacker tool after </think> (benign rows: calls recorded).
Output: results/e093-cloud/{run.json, intervene.jsonl}; uploaded to the artifact repo under e093/<run_id>/.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_states import render, sha, snapshot, verify
from injecforks_resample import expand, label, sample


class Steer:
    """Adds rows of `self.v` [B, d] to a decoder layer's output while `self.v` is set; read positions are untouched otherwise."""

    def __init__(self, layer_mod):
        self.v = None; self.h = layer_mod.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        if self.v is None:
            return None
        if isinstance(out, (tuple, list)):
            return type(out)((out[0] + self.v[:, None, :].to(out[0].dtype), *out[1:]))
        return out + self.v[:, None, :].to(out.dtype)


def run(job, smoke=False):
    import torch
    from common import load_generator
    assert torch.cuda.is_available()
    api = None
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), steer_layer=job["smoke_steer_layer"], **job["smoke_overrides"])
        run_id = "smoke"
    else:
        from huggingface_hub import HfApi
        api = HfApi()
        if not api.repo_info(job["artifact_repo"]).private:
            raise SystemExit("artifact repo must be private")
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(dict(job, select={"realistic": ["dev"], "realistic-benign": ["dev"]})); nver = verify(root)
    RES = HERE / "results/e093-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp = job["sampling"]; NR, MAXN = job["n"], job["max_new_tokens"]
    d_model = model.lm_head.weight.shape[1]
    if smoke:   # the 9B direction has the wrong width; any unit vectors exercise the path
        g = np.random.default_rng(0); u = g.standard_normal(d_model); u /= np.linalg.norm(u); r = g.standard_normal(d_model); r /= np.linalg.norm(r)
        gap = 1.0
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(job["artifact_repo"], job["steer"]["path"])
        assert sha(Path(p)) == job["steer"]["sha256"], "steering vectors differ from the config"
        z = np.load(p); u, r, gap = z["u"], z["rand"], float(z["class_gap"])
        assert u.shape == (d_model,), (u.shape, d_model)
    dirs = {"probe": torch.tensor(u, device="cuda"), "random": torch.tensor(r, device="cuda")}
    arms = {k: v for k, v in job["arms"].items()}   # name -> {direction, alpha_gaps} (steer) or {} (control)
    rem = tok(job["reminder"], add_special_tokens=False).input_ids
    steer = Steer(layers[job["steer_layer"] - 1])
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, started=time.time(), verified_files=nver, n=NR, max_new_tokens=MAXN,
                   sampling=sp, arms=arms, reminder=job["reminder"], reminder_tokens=len(rem), class_gap=gap, cases_done=0)
    fout = open(RES / "intervene.jsonl", "w" if smoke else "a"); last = time.time()

    def flush(final=False):
        fout.flush()
        receipt.update(uploaded=time.time(), final=final, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e093/{run_id}/results")

    def vec(arm, B):
        a = arms[arm]
        if not a:
            return torch.zeros(B, d_model, device="cuda")
        return (-a["alpha_gaps"] * gap * dirs[a["direction"]]).expand(B, -1)

    try:
        work = []
        for cfg, per_case in (("realistic-benign", job["benign_rollouts"]), ("realistic", None)):   # benign first: a timer kill keeps it
            ev = {(e["case"], e["rollout"]) for e in map(json.loads, open(root / "eval-v1.2" / cfg / "dev.jsonl"))}
            cuts = {(c["case"], c["rollout"]): c for c in map(json.loads, open(root / "cuts" / cfg / "dev.jsonl"))}
            by = {}
            for l in open(root / cfg / "dev.jsonl"):
                r_ = json.loads(l)
                if (r_["case"], r_["rollout"]) in ev:
                    by.setdefault(r_["case"], []).append(r_)
            bodies = {b["case"]: b for b in map(json.loads, open(root / "prompts" / cfg / "dev.jsonl")) if b["case"] in by}
            for case in sorted(by)[:job.get("max_cases") or None]:
                work.append((cfg, case, bodies[case], sorted(by[case], key=lambda x: x["rollout"])[:per_case], cuts))
        t0 = time.time()
        for wi, (cfg, case, b, rs, cuts) in enumerate(work):
            pids = render(tok, b)
            assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case, len(pids), b["prompt_tokens"])
            P = len(pids); atk = rs[0]["attacker_tools"]
            gen = torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e093|{cfg}|{case}".encode()).hexdigest()[:8], 16))
            with torch.no_grad():
                o = lm(input_ids=torch.tensor([pids], device="cuda"), use_cache=True, position_ids=torch.arange(P, device="cuda")[None])
            base = o.past_key_values; V = model.lm_head.weight.shape[0]
            pre = []
            for r_ in rs:
                ids = tok(r_["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
                c = cuts[(case, r_["rollout"])]["cut"].get(str(NR))
                if c is None:
                    continue
                assert ids.offset_mapping[NR - 1][1] == c and think_end not in ids.input_ids[:NR], (case, r_["rollout"])
                pre.append((r_, ids.input_ids[:NR]))
            plan = job["plan"]["injected" if cfg == "realistic" else "benign"]   # arm -> samples per rollout
            # one batch of plain-prefix arms (control and steering), one of reminder continuations
            groups = [("plain", [(a, k) for a, k in plan.items() if a != "reminder"]), ("reminder", [("reminder", plan.get("reminder", 0))])]
            for kind, arm_k in groups:
                rows = [(r_, p, a, j) for r_, p in pre for a, k in arm_k for j in range(k)]
                for s in range(0, len(rows), job["batch"]):
                    ch = rows[s:s + job["batch"]]; B = len(ch)
                    seqs = [p + (rem if kind == "reminder" else []) for _, p, _, _ in ch]; Lp = len(seqs[0])
                    x = torch.tensor(seqs, device="cuda")
                    with torch.no_grad():
                        o2 = lm(input_ids=x, past_key_values=expand(base, B), use_cache=True,
                                position_ids=torch.arange(P, P + Lp, device="cuda")[None].expand(B, -1))
                    if wi == 0 and s == 0 and kind == "plain":   # first case (benign); same check as E090
                        with torch.no_grad():
                            hf = lm(input_ids=torch.tensor([pids + seqs[-1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                        cos = float(torch.nn.functional.cosine_similarity(o2.last_hidden_state[-1, -1].float(), hf, 0))
                        receipt["cache_check_cosine"] = round(cos, 5)
                        assert cos > 0.995, ("copied cache differs from a full forward", case, cos)
                    seen = torch.zeros(B, V, dtype=torch.bool, device="cuda"); seen[torch.arange(B)[:, None], x] = True
                    # the first sampled token comes from o2's logits (unsteered read of the prefix); steering acts from then on
                    steer.v = torch.stack([vec(a, 1)[0] for _, _, a, _ in ch]) if kind == "plain" else None
                    try:
                        outs = sample(model, lm, o2.past_key_values, model.lm_head(o2.last_hidden_state[:, -1]), P + Lp, seen, sp, eos, MAXN, gen)
                    finally:
                        steer.v = None
                    for (r_, p, a, j), g, sq in zip(ch, outs, seqs):
                        txt = tok.decode(sq + g, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                        fout.write(json.dumps(dict(config=cfg, case=case, rollout=r_["rollout"], arm=a, j=j, orig_violates=r_["violates"],
                                                   orig_strict=r_["violates_strict"], decided=dec, violates_strict=v, calls=calls,
                                                   new_tokens=len(g), text=txt)) + "\n")
                    del o2
            del base, o; torch.cuda.empty_cache()
            receipt["cases_done"] = wi + 1
            print(wi + 1, len(work), cfg, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["seconds"] = round(time.time() - t0, 1); receipt["complete"] = True
    finally:
        receipt["finished"] = time.time(); flush(final=True); fout.close()
    print("done", json.dumps({k: receipt.get(k) for k in ("cases_done", "seconds", "peak_gib")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e093/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e093/{prefix}/*"], local_dir=st)
    print("fetched to", st / "e093" / prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["smoke", "run", "fetch"]); ap.add_argument("--prefix")
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"smoke": lambda: run(job, smoke=True), "run": lambda: run(job), "fetch": lambda: fetch(job, a.prefix)}[a.stage]()
