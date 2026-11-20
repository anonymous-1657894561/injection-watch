"""E097: the intervention arms, once on test (decisions/, 2026-10-30 "E097: the intervention test protocol"). Qwen3.5-9B BF16.

  ~/.venvs/e01/bin/python injecforks_e097.py configs/e097-9b-test-pod.yaml dry [--split dev|test]   # CPU: tokenizer only, no model
  ~/.venvs/e01/bin/python injecforks_e097.py configs/e097-9b-test-pod.yaml smoke                    # local GPU: Qwen3.5-0.8B, short
  python injecforks_e097.py configs/e097-9b-test-pod.yaml run --split test --final                  # pod, once (testread.py)
  ~/.venvs/e01/bin/python injecforks_e097.py configs/e097-9b-test-pod.yaml fetch --prefix RUN_ID
  ~/.venvs/e01/bin/python injecforks_e097.py configs/e097-9b-test-pod.yaml score --prefix RUN_ID    # delegation replies -> ensemble scores

Three parts with E090's sampler, most important first (a timer kill keeps what is done):
(a) rows: realistic eval rows of the split whose reasoning reaches N = 16 (test: rollouts 8-15 only), each continued from its
    first 16 reasoning tokens under control, reminder, placebo, probe α8 and random α8 k = 0..4 (k = 0 is E093's random direction,
    k = 1..4 E096's, numpy default_rng(1000 + k) orthogonal to u); realistic-benign eval rows under control, reminder, placebo
    and probe α8. Steering adds -8 x class gap x direction to layer 20's output at every generated token (the prefix read is
    unsteered), as injecforks_intervene.py. Benign rows go first in each case batch order, as E093.
(b) N = 0: per realistic case, k0 samples from the prompt with the reminder as the first reasoning tokens and k0 without;
    realistic-benign cases, 4 and 4 (injecforks_e096.py part (c), with the control sampled here rather than taken from E090).
(c) delegation, the held-out half of results/e096/cases.jsonl: injecforks_e096.py part (a) unchanged (K replies per version,
    layer-20 states at N = 0 and each read, self-report at N = 0 and 16, continuations under control, reminder, placebo).
Records are in injecforks_intervene.py's and injecforks_e096.py's formats, so e097_analyze.py reads them unchanged. Only case,
rollout, reasoning and attacker tools are kept from the rollout files; no dataset label is read into a record (e097_analyze.py
joins them). Output: results/e097-cloud/{run.json, e097.jsonl, states.npy, states-meta.jsonl}; uploaded under e097/<run_id>/.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_states import render, sha, snapshot, verify
import testread

E093_RUN = "e093/user-e093-9b-pod-20261027T200233Z/results"
E096_RUN = "e096/user-e096-9b-pod-20261029T190638Z/results"
NAME = "e097-arms"


def frozen(job):
    """The protocol's fixed numbers, asserted against the configs the dev arms were run with (E090, E093, E096)."""
    e090, e093, e096 = (yaml.safe_load((HERE / f"configs/{c}-9b-pod.yaml").read_text()) for c in ("e090", "e093", "e096"))
    assert job["sampling"] == e090["sampling"] == e093["sampling"] == e096["sampling"], "E090's sampler"
    assert job["max_new_tokens"] == 1536 and job["n"] == 16 and job["state_layer"] == job["steer_layer"] == 20
    assert job["alpha"] == 8 and job["random_seeds"] == [1, 2, 3, 4] and job["steer"] == e093["steer"] == e096["steer"]
    assert job["steer"]["sha256"].startswith("99ead71b")
    assert (job["reminder"], job["placebo"], job["reminder_n0"]) == (e096["reminder"], e096["placebo"], e096["reminder_n0"])
    assert job["reminder"] == e093["reminder"] and job["generator"] == e096["generator"] and job["dataset_revision"] == e096["dataset_revision"]
    assert sha(HERE / "results/e096/cases.jsonl").startswith("859a5469")


def inserts(tok, job):
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    rem, plc, rem0 = enc(job["reminder"]), enc(job["placebo"]), enc(job["reminder_n0"])
    assert len(enc(job["reminder"].strip())) == len(enc(job["placebo"].strip())) == 39 and len(rem) == len(plc), (len(rem), len(plc))
    return rem, plc, rem0


def rows(root, split, cfg, NR, per_case, max_cases=None):
    """[(case, body, [(rollout, reasoning, attacker_tools, cut)])] of the eval rows that reach NR reasoning tokens (NR None: every
    eval row; the N = 0 part samples per case); realistic test: rollouts 8-15. Label columns are dropped as each rollout is read."""
    ev = {(e["case"], e["rollout"]) for e in map(json.loads, open(root / "eval-v1.2" / cfg / f"{split}.jsonl"))}
    cuts = {(c["case"], c["rollout"]): c["cut"].get(str(NR or 0)) for c in map(json.loads, open(root / "cuts" / cfg / f"{split}.jsonl"))}
    by = {}
    for l in open(root / cfg / f"{split}.jsonl"):
        r = json.loads(l); k = (r["case"], r["rollout"])
        if k in ev and (NR is None or cuts[k] is not None) and (NR is None or split == "dev" or cfg != "realistic" or r["rollout"] >= 8):
            by.setdefault(r["case"], []).append((r["rollout"], r["reasoning"], r["attacker_tools"], cuts[k]))
    bodies = {b["case"]: b for b in map(json.loads, open(root / "prompts" / cfg / f"{split}.jsonl")) if b["case"] in by}
    return [(case, bodies[case], sorted(by[case])[:per_case]) for case in sorted(by)[:max_cases or None]]


def prefixes(tok, rs, NR, think_end):
    out = []
    for rollout, reasoning, _, cut in rs:
        e = tok(reasoning, add_special_tokens=False, return_offsets_mapping=True)
        assert e.offset_mapping[NR - 1][1] == cut and think_end not in e.input_ids[:NR], rollout   # the dataset's cut, as E093
        out.append((rollout, e.input_ids[:NR]))
    return out


def plan(job, cfg):
    """Arm -> inserted ids key (None: none) and steering direction (None: unsteered)."""
    steer = {"probe_a8": "probe", **({f"random{k}_a8": k for k in range(5)} if cfg == "realistic" else {})}
    return [("control", None, None), ("reminder", "rem", None), ("placebo", "plc", None)] + [(a, None, d) for a, d in steer.items()]


def directions(job, d_model, smoke):
    """u, class gap and the five random directions (k = 0: E093's; k = 1..4: E096's), from the artifact repo."""
    if smoke:   # the 9B direction has the wrong width; any unit vectors exercise the path
        g = np.random.default_rng(0); u = g.standard_normal(d_model); u /= np.linalg.norm(u); r0 = g.standard_normal(d_model); r0 /= np.linalg.norm(r0)
        gap = 1.0
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(job["artifact_repo"], job["steer"]["path"])
        assert sha(Path(p)) == job["steer"]["sha256"], "steering vectors differ from the config"
        z = np.load(p); u, r0, gap = z["u"], z["rand"], float(z["class_gap"])
        assert u.shape == (d_model,) and abs(gap - 2.463) < 5e-4, (u.shape, gap)
    R = {0: r0}
    for k in job["random_seeds"]:
        r = np.random.default_rng(1000 + k).standard_normal(d_model); r -= (r @ u) * u; R[k] = r / np.linalg.norm(r)
    return u, gap, R


def dry(job, split):
    """CPU, no model: builds every part's work list and, on dev, checks the inputs against E093's and E096's stored dev records
    (each continuation's text starts with the decoded prefix + insert this script would send; random directions equal E096's)."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from e096_delegation import bodies
    frozen(job)
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    rem, plc, rem0 = inserts(tok, job); ins = dict(rem=rem, plc=plc); think_end = tok.convert_tokens_to_ids("</think>"); NR = job["n"]
    root = snapshot(dict(job, select={"realistic": sorted({"dev", split}), "realistic-benign": [split]})); verify(root)
    out = {}
    for cfg in ("realistic-benign", "realistic"):
        W = rows(root, split, cfg, NR, job["benign_rollouts"] if cfg != "realistic" else None)
        n_rows = sum(len(rs) for _, _, rs in W)
        n0c = len(rows(root, split, cfg, None, None))
        out[cfg] = dict(cases=len(W), rows=n_rows, sequences=n_rows * len(plan(job, cfg)),
                        n0_cases=n0c, n0_sequences=n0c * 2 * (job["k0"] if cfg == "realistic" else job["benign_rollouts"]))
        if split == "dev":
            for case, b, rs in W[:2]:
                assert len(render(tok, b)) == b["prompt_tokens"], case
    D = bodies("heldout", root)
    out["delegation"] = dict(bodies=len(D), sequences=len(D) * job["k"] * 4)   # K replies, then three continuations of each
    for case, ver, body in D:
        if ver == "unauth":
            assert len(render(tok, body)) == body["prompt_tokens"], case
    out["sequences_total"] = sum(v["sequences"] + v.get("n0_sequences", 0) for v in out.values() if isinstance(v, dict))
    if split == "dev":   # inputs equal what E093 / E096 sent for the same rows
        f93 = hf_hub_download(job["artifact_repo"], f"{E093_RUN}/intervene.jsonl"); f96 = hf_hub_download(job["artifact_repo"], f"{E096_RUN}/e096.jsonl")
        ref = {}
        for f in (f93, f96):
            for x in map(json.loads, open(f)):
                if "rollout" in x and x.get("arm") in ("control", "reminder", "placebo", "probe_a8", "random_a8", "random1_a8"):
                    ref.setdefault((x["config"], x["case"], x["rollout"], x["arm"]), x["text"])
        n = 0
        for cfg in ("realistic-benign", "realistic"):
            for case, b, rs in rows(root, "dev", cfg, NR, job["benign_rollouts"] if cfg != "realistic" else None, max_cases=2):
                for rollout, p in prefixes(tok, rs, NR, think_end):
                    for arm, k, _ in plan(job, cfg):
                        t = ref.get((cfg, case, rollout, "random_a8" if arm == "random0_a8" else arm))
                        if t is not None:
                            assert t.startswith(tok.decode(p + (ins[k] if k else []), skip_special_tokens=False)), (cfg, case, rollout, arm); n += 1
        x0 = next(x for x in map(json.loads, open(f96)) if x.get("part") == "reminder_n0" and x["arm"] == "reminder_n0")
        assert x0["text"].startswith(tok.decode(rem0, skip_special_tokens=False)); n += 1
        z = np.load(hf_hub_download(job["artifact_repo"], job["steer"]["path"])); _, gap, R = directions(job, z["u"].shape[0], smoke=False)
        zr = np.load(hf_hub_download(job["artifact_repo"], f"{E096_RUN}/random_dirs.npz"))
        # equal to rounding (the projection's dot product differs by ~1e-18 across BLAS builds)
        assert all(np.allclose(R[k], zr[f"r{k}"], rtol=0, atol=1e-12) for k in job["random_seeds"]) and np.array_equal(R[0], z["rand"]), "random directions"
        out["dev_checks"] = dict(inputs_matching_e093_e096=n, random_directions="equal E093's and E096's", class_gap=round(gap, 4))
    print(json.dumps(out, indent=1))


def run(job, split, final, smoke=False):
    import torch
    from common import LayerTap, load_generator
    from e096_delegation import bodies
    from injecforks_intervene import Steer
    from injecforks_resample import expand, label, sample
    from injecforks_selfreport import FMT, Q, answer_ids
    if not smoke:
        testread.check(argparse.Namespace(split=split, final=final), NAME)
    assert torch.cuda.is_available()
    frozen(job); api = None
    if smoke:
        job = dict(job, generator=dict(job["smoke_generator"]), state_layer=job["smoke_state_layer"], steer_layer=job["smoke_state_layer"],
                   **job["smoke_overrides"])
        run_id = "smoke"
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
                if f.startswith("e097/") and f.endswith("results/run.json"):
                    rj = json.load(open(hf_hub_download(job["artifact_repo"], f)))
                    if rj.get("split") == "test" and rj.get("complete"):
                        raise SystemExit(f"test arms were already run ({f}); a second run is not the pre-registered result")
        run_id = os.environ.get("RPJOB_RUN_ID") or time.strftime("%Y%m%dT%H%M%S")
    root = snapshot(dict(job, select={"realistic": sorted({"dev", split}), "realistic-benign": [split]})); nver = verify(root)
    RES = HERE / "results/e097-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    model, tok, layers = load_generator(job["generator"]); lm = model.model.language_model
    eos = tok.convert_tokens_to_ids("<|im_end|>"); think_end = tok.convert_tokens_to_ids("</think>")
    sp, MAXN, NR, L = job["sampling"], job["max_new_tokens"], job["n"], job["state_layer"]
    rem, plc, rem0 = inserts(tok, job); ins = dict(rem=rem, plc=plc)
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    ans = answer_ids(tok); yes, no = torch.tensor(ans["yes"], device="cuda"), torch.tensor(ans["no"], device="cuda")
    qsr = enc(FMT["turn"].format(q=Q["comply"]))
    d_model = model.lm_head.weight.shape[1]; V = model.lm_head.weight.shape[0]
    u, gap, R = directions(job, d_model, smoke)
    vec = {"probe": torch.tensor(-job["alpha"] * gap * u, device="cuda", dtype=torch.float32)[None],
           **{k: torch.tensor(-job["alpha"] * gap * r, device="cuda", dtype=torch.float32)[None] for k, r in R.items()}}
    receipt = dict(gpu=torch.cuda.get_device_name(), run_id=run_id, split=split, started=time.time(), verified_files=nver, sampling=sp,
                   max_new_tokens=MAXN, n=NR, alpha=job["alpha"], class_gap=gap, reminder_tokens=len(rem), placebo_tokens=len(plc),
                   random_dirs={k: dict(cos_u=round(float(r @ u), 6)) for k, r in R.items()}, state_layer=L, parts_done=[], cache_check=[])
    fout = open(RES / "e097.jsonl", "w" if smoke else "a"); SX, SM = [], []; last = time.time(); grab = {}
    tap = LayerTap(layers, [L], lambda li, h: grab.__setitem__("h", h))   # [B, T, d] output of layer L on the latest forward
    steer = Steer(layers[job["steer_layer"] - 1])

    def flush(final_=False):
        fout.flush()
        if SX:
            np.save(RES / "states.npy", np.stack(SX).astype(np.float16))
            (RES / "states-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in SM))
        receipt.update(uploaded=time.time(), final=final_, peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"e097/{run_id}/results")

    def fwd(ids, cache, start, B=1):
        x = ids if torch.is_tensor(ids) else torch.tensor(ids, device="cuda")
        x = x[None].expand(B, -1) if x.dim() == 1 else x
        with torch.no_grad():
            return lm(input_ids=x, past_key_values=cache, use_cache=True,
                      position_ids=torch.arange(start, start + x.shape[1], device="cuda")[None].expand(x.shape[0], -1))

    def prefill(pids):
        o = fwd(pids, None, 0); return o.past_key_values, model.lm_head(o.last_hidden_state[:, -1]), grab["h"][0, -1].float().cpu().numpy()

    def selfreport(cache, start):
        with torch.no_grad():
            o = lm(input_ids=torch.tensor([qsr], device="cuda"), past_key_values=expand(cache, 1), use_cache=True,
                   position_ids=torch.arange(start, start + len(qsr), device="cuda")[None])
        p = torch.softmax(model.lm_head(o.last_hidden_state[0, -1]).float(), -1); py, pn = p[yes].sum(), p[no].sum()
        return round(float(py / (py + pn)), 6), round(float(py + pn), 6)

    def gen(tag):
        return torch.Generator(device="cuda").manual_seed(int(hashlib.sha256(f"e097|{split}|{tag}".encode()).hexdigest()[:8], 16))

    def write(**kw):
        fout.write(json.dumps(kw) + "\n")

    def continue_from(base, P, pre, arms, g):
        """pre: [(meta, ids)]; arms: [(name, inserted ids, steering vector or None)]. Every prefix under every arm, batched."""
        for name, add, v in arms:
            rows_ = [(m, p + add) for m, p in pre]
            for s in range(0, len(rows_), job["batch"]):
                ch = rows_[s:s + job["batch"]]; B = len(ch); x = torch.tensor([q for _, q in ch], device="cuda")
                o2 = fwd(x, expand(base, B), P, B)
                if not receipt["cache_check"]:   # the copied cache equals one full forward (as E090 / E093)
                    with torch.no_grad():
                        hf = lm(input_ids=torch.tensor([pids_now[0] + ch[-1][1]], device="cuda"), use_cache=False).last_hidden_state[0, -1].float()
                    cos = float(torch.nn.functional.cosine_similarity(o2.last_hidden_state[-1, -1].float(), hf, 0))
                    receipt["cache_check"].append(round(cos, 5)); assert cos > job.get("min_cache_cosine", 0.99), ("copied cache differs from a full forward", cos)  # decisions/, 2026-11-02 and 11-05
                seen = torch.zeros(B, V, dtype=torch.bool, device="cuda"); seen[torch.arange(B)[:, None], x] = True
                steer.v = v.expand(B, -1) if v is not None else None
                try:
                    outs = sample(model, lm, o2.past_key_values, model.lm_head(o2.last_hidden_state[:, -1]), P + x.shape[1], seen, sp, eos, MAXN, g)
                finally:
                    steer.v = None
                for (m, q), o in zip(ch, outs):
                    yield m, name, tok.decode(q + o, skip_special_tokens=False), len(o)
                del o2

    pids_now = [None]
    try:
        # ---- (a) rows at N = 16 ----
        work = [(cfg, *w) for cfg in ("realistic-benign", "realistic")
                for w in rows(root, split, cfg, NR, job["benign_rollouts"] if cfg != "realistic" else None, job.get("max_cases"))]
        t0 = time.time()
        for wi, (cfg, case, b, rs) in enumerate(work):
            pids = render(tok, b); assert len(pids) == b["prompt_tokens"], ("billed prompt tokens differ", case)
            pids_now[0] = pids; P = len(pids); atk = rs[0][2]; base, _, _ = prefill(pids); g = gen(f"rows|{cfg}|{case}")
            arms = [(a, ins[k] if k else [], vec[d] if d is not None else None) for a, k, d in plan(job, cfg)]
            for rollout, arm, txt, n_new in continue_from(base, P, prefixes(tok, rs, NR, think_end), arms, g):
                dec, v, calls = label(txt, atk)
                write(config=cfg, case=case, rollout=rollout, arm=arm, j=0, decided=dec, violates_strict=v, calls=calls, new_tokens=n_new, text=txt)
            del base; torch.cuda.empty_cache()
            print("a", wi + 1, len(work), cfg, f"{time.time() - t0:.0f}s", f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["parts_done"].append("rows"); flush(); last = time.time()

        # ---- (b) N = 0: the reminder as the first reasoning tokens, and plain samples from the prompt ----
        work = [(cfg, *w) for cfg in ("realistic-benign", "realistic") for w in rows(root, split, cfg, None, None, job.get("max_cases"))]
        t0 = time.time()
        for wi, (cfg, case, b, rs) in enumerate(work):
            pids = render(tok, b); pids_now[0] = pids; P = len(pids); atk = rs[0][2]; base, first, _ = prefill(pids); g = gen(f"n0|{cfg}|{case}")
            n0 = job["k0"] if cfg == "realistic" else job["benign_rollouts"]
            outs = sample(model, lm, expand(base, n0), first.repeat(n0, 1), P, torch.zeros(n0, V, dtype=torch.bool, device="cuda"), sp, eos, MAXN, g)
            res = [("control_n0", tok.decode(o, skip_special_tokens=False), len(o)) for o in outs]
            res += [(a, t, n) for _, a, t, n in continue_from(base, P, [(None, [])] * n0, [("reminder_n0", rem0, None)], g)]
            for j, (arm, txt, n_new) in enumerate(res):
                dec, v, calls = label(txt, atk)
                write(part="reminder_n0", config=cfg, case=case, arm=arm, j=j % n0, decided=dec, violates_strict=v, calls=calls, new_tokens=n_new, text=txt)
            del base; torch.cuda.empty_cache()
            print("b", wi + 1, len(work), cfg, f"{time.time() - t0:.0f}s", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["parts_done"].append("n0"); flush(); last = time.time()

        # ---- (c) delegation, held-out half (injecforks_e096.py part (a)) ----
        cases = {x["case"]: x for x in map(json.loads, open(HERE / "results/e096/cases.jsonl"))}
        work = bodies("heldout", root)[:job.get("max_delegation_bodies") or None]
        t0 = time.time()
        for wi, (case, ver, body) in enumerate(work):
            pids = render(tok, body); pids_now[0] = pids; P = len(pids); atk = cases[case]["attacker_tools"]
            if ver == "unauth":
                assert P == body["prompt_tokens"], ("billed prompt tokens differ", case, P, body["prompt_tokens"])
            base, first, s0 = prefill(pids); K = job["k"]; g = gen(f"deleg|{case}|{ver}")
            rid0 = f"deleg|{ver}|{case}"; SX.append(s0); SM.append(dict(row_id=rid0, N=0))
            sr0 = selfreport(base, P)
            outs = sample(model, lm, expand(base, K), first.repeat(K, 1), P, torch.zeros(K, V, dtype=torch.bool, device="cuda"), sp, eos, MAXN, g)
            pre = []
            for k, o in enumerate(outs):
                txt = tok.decode(o, skip_special_tokens=False); dec, v, calls = label(txt, atk)
                reasoning = txt.split("</think>", 1)[0]; e = tok(reasoning, add_special_tokens=False, return_offsets_mapping=True)
                rid = f"{rid0}|{k}"; reads = [n for n in job["reads"] if len(e.input_ids) >= n]; cut = {n: e.offset_mapping[n - 1][1] for n in reads}
                rec = dict(part="delegation", kind="reply", case=case, version=ver, reply=k, row_id=rid, decided=dec, violates_strict=v, calls=calls,
                           new_tokens=len(o), reasoning_tokens=len(e.input_ids), cut=cut, text=txt, selfreport={"0": sr0})
                if reads:
                    ids = e.input_ids[:max(reads)]; assert think_end not in ids
                    c = copy.deepcopy(base); pos = 0
                    for n in reads:
                        o2 = fwd(ids[pos:n], c, P + pos); c = o2.past_key_values; pos = n
                        SX.append(grab["h"][0, -1].float().cpu().numpy()); SM.append(dict(row_id=rid, N=n))
                        if n == NR:
                            rec["selfreport"][str(NR)] = selfreport(c, P + n)
                    if NR in reads:
                        pre.append((dict(rid=rid, k=k), ids[:NR]))
                write(**rec)
            for m, arm, txt, n_new in continue_from(base, P, pre, [("control", [], None), ("reminder", rem, None), ("placebo", plc, None)], g):
                dec, v, calls = label(txt, atk)
                write(part="delegation", kind="continuation", case=case, version=ver, reply=m["k"], row_id=m["rid"], arm=arm, decided=dec,
                      violates_strict=v, calls=calls, new_tokens=n_new, text=txt)
            del base; torch.cuda.empty_cache()
            print("c", wi + 1, len(work), f"{time.time() - t0:.0f}s", flush=True)
            if time.time() - last > job.get("upload_every_s", 600):
                flush(); last = time.time()
        receipt["parts_done"].append("delegation"); receipt["complete"] = True
    finally:
        tap.close(); steer.h.remove(); receipt["finished"] = time.time(); flush(final_=True); fout.close()
    if not smoke and receipt.get("complete"):
        testread.done(argparse.Namespace(split=split, final=final), NAME, run_id=run_id)
    print("done", json.dumps({k: receipt.get(k) for k in ("parts_done", "peak_gib", "cache_check")}))


def fetch(job, prefix):
    from huggingface_hub import snapshot_download
    st = HERE / "data/e097/dl"; snapshot_download(job["artifact_repo"], allow_patterns=[f"e097/{prefix}/*"], local_dir=st)
    print("fetched to", st / "e097" / prefix)


def score(prefix):
    """The delegation replies' layer-20 states -> the frozen 5-seed comply ensemble's scores ({row_id, N, score}), the
    --deleg-scores input of e097_analyze.py. CPU; refits the ensemble on train (e096_analyze.ensemble, asserted on dev)."""
    from e096_analyze import ensemble
    R = HERE / "data/e097/dl/e097" / prefix / "results"
    X = np.load(R / "states.npy").astype(np.float32); M = [json.loads(l) for l in open(R / "states-meta.jsonl")]; assert len(X) == len(M)
    f, _ = ensemble(); s = f(X); out = R / "deleg-scores.jsonl"
    out.write_text("".join(json.dumps(dict(row_id=m["row_id"], N=m["N"], score=float(v))) + "\n" for m, v in zip(M, s)))
    print("wrote", out, len(M))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["dry", "smoke", "run", "fetch", "score"])
    ap.add_argument("--prefix"); testread.args(ap)
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"dry": lambda: dry(job, a.split), "smoke": lambda: run(job, "dev", False, smoke=True), "run": lambda: run(job, a.split, a.final),
     "fetch": lambda: fetch(job, a.prefix), "score": lambda: score(a.prefix)}[a.stage]()
