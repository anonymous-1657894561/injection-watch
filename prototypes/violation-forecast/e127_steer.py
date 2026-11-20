"""E127: Qwen3.5-9B on InjecForks realistic test cases, steered at layer 20 as the paper's Method (experiments/E127-comply-steering.md).

  ~/.venvs/e01/bin/python e127_steer.py configs/e127-9b-steer.yaml prep    # local, CPU: case draw, prompts and prefixes checked
  ~/.venvs/e01/bin/python e127_steer.py configs/e127-9b-steer.yaml smoke   # local GPU: 0.8B, 1 case, hook, cache and sampler checks
  python e127_steer.py configs/e127-9b-steer.yaml run                      # pod: the first run_cases cases of the draw

Cases: realistic test cases whose 8 eval-v1.2 rollouts called the attacker tool >= min_eval_strict times, drawn with case_seed;
a run takes the first run_cases of the draw and skips cases already on the artifact repo with the same arms.
Prefixes: each case's other 8 rollouts (evalextra, not used to pick the case) that reach N = 16 reasoning tokens; every arm
continues the same first 16 reasoning tokens of the same rollouts (an arm with k = 4 takes the first 4), so arms are paired by
prefix. Prompts: the dataset's request bodies, rendered with injecforks_states.render and asserted equal to the billed prompt_tokens.
Steering: h <- h + alpha * gap * d on the OUTPUT of decoder layer 20 (1-indexed), on every generated token after the prefix (the
prompt and the prefix are read unsteered). d = E093's u (the paper's probe direction) or a random unit direction orthogonal to it
(k = 0: E093's; k >= 1: numpy default_rng(1000 + k), as E096 / E097); gap = E093's class gap (2.463). Away = negative alpha.
Sampler: the dataset's (presence penalty on reasoning and answer tokens, prefix included, then temperature, top-k, top-p, vLLM's
order), as a logits processor with HF's own warpers switched off; max tokens as the dataset's, prefix included.
Labels: strict and broad as e112_rollouts.rows_of, on the prefix plus the continuation.
Output: data/<out_name>/rows.jsonl (+ results/e127-cloud/run.json), uploaded to artifact_repo under out_name/.
"""
import argparse, hashlib, json, os, random, re, sys, time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from injecforks_states import render, sha, verify
from e112_rollouts import CASE_COLS, rows_of

REPO = "anonymous/InjecForks"


def case_seed(case):
    return int(hashlib.sha256(f"e127|{case}".encode()).hexdigest(), 16) % 2**31


def arms_sha(job):
    return hashlib.sha256(json.dumps(job["arms"], sort_keys=True).encode()).hexdigest()[:12]


def draw(job):
    """[(case, dataset strict rate on its eval rows)] in draw order. Cases where the dataset's 8 eval-v1.2 rollouts called the
    attacker tool at least min_eval_strict times (room to steer compliance down); evalextra stays held out as the baseline check."""
    from injecforks_load import load
    by = {}
    for r in load("realistic", "test"):   # eval-v1.2 rows, labels v1.2
        by.setdefault(r["case"], []).append(int(r["violates_strict"]))
    pool = sorted(c for c, v in by.items() if sum(v) >= job["min_eval_strict"])
    pick = random.Random(job["case_seed"]).sample(pool, job["n_cases"])
    return [(c, sum(by[c]) / len(by[c])) for c in pick], len(pool)


def items_of(job, tok, cases):
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(REPO, repo_type="dataset", revision=job["dataset_revision"],
                                  allow_patterns=["realistic/test.jsonl", "prompts/realistic/test.jsonl", "SHA256SUMS*"]))
    verify(root)
    want = {c for c, _ in cases}; meta, bodies = {}, {}
    for l in open(root / "realistic/test.jsonl"):
        r = json.loads(l)
        if r["case"] in want:
            meta.setdefault(r["case"], {k: r[k] for k in CASE_COLS if k in r})
    for l in open(root / "prompts/realistic/test.jsonl"):
        c = re.match(r'\{"case": "([^"]+)"', l).group(1)
        if c in want:
            bodies[c] = json.loads(l)
    assert set(bodies) == set(meta) == want, "prompts or rows missing"
    from injecforks_load import load
    ex = {}
    for r in load("realistic", "test", subset="evalextra"):
        if r["case"] in want and r["cut"].get(job["n"]) is not None:
            ex.setdefault(r["case"], []).append(r)
    think_end = tok.convert_tokens_to_ids("</think>")
    out = []
    for c, rate in cases:
        ids = render(tok, bodies[c])
        assert len(ids) == bodies[c]["prompt_tokens"], ("prompt tokens differ from the dataset's", c, len(ids))
        pre = []
        for r in sorted(ex.get(c, []), key=lambda r: int(r["rollout"])):
            e = tok(r["reasoning"], add_special_tokens=False, return_offsets_mapping=True)
            assert e.offset_mapping[job["n"] - 1][1] == r["cut"][job["n"]] and think_end not in e.input_ids[:job["n"]], (c, r["rollout"])
            pre.append((int(r["rollout"]), e.input_ids[:job["n"]], int(r["violates_strict"])))
        out.append((meta[c], bodies[c], ids, rate, pre))
    return out


class Sampler:
    """vLLM's order on one step: presence penalty (generated tokens only), temperature, top-k, top-p."""

    def __init__(self, P, body):
        self.P, self.pres, self.t, self.k, self.p = P, body["presence_penalty"], body["temperature"], body["top_k"], body["top_p"]

    def __call__(self, input_ids, scores):
        import torch
        s = scores.float()
        gen = input_ids[:, self.P:]
        if self.pres and gen.shape[1]:
            seen = torch.zeros_like(s, dtype=torch.bool).scatter_(1, gen, True)
            s = s - self.pres * seen
        s = s / self.t
        if self.k and self.k > 0:
            s = s.masked_fill(s < torch.topk(s, self.k, dim=-1).values[:, -1:], float("-inf"))
        if self.p < 1:
            srt, idx = torch.sort(s, descending=True, dim=-1)
            cum = torch.softmax(srt, -1).cumsum(-1)
            drop = cum - torch.softmax(srt, -1) > self.p          # keep the smallest prefix whose mass reaches top_p
            s = s.scatter(1, idx, srt.masked_fill(drop, float("-inf")))
        return s


class Steer:
    """Adds self.v[b] to row b of a decoder layer's output on decode steps only. self.v = None: a no-op."""

    def __init__(self, layer):
        self.v, self.edits, self.skip = None, 0, 0
        self.h = layer.register_forward_hook(self.hook)

    def hook(self, _m, _inp, out):
        if self.v is None:
            return None
        h = out[0] if isinstance(out, (tuple, list)) else out
        if h.shape[1] != 1:   # a prefill: the prompt is not edited
            return None
        if self.skip:         # generate's first step re-feeds the last prompt token: not edited either
            self.skip -= 1
            return None
        h = h + self.v.to(h.dtype)
        self.edits += 1
        return (h, *out[1:]) if isinstance(out, (tuple, list)) else h


def generate(model, tok, steer, ids, body, vecs, max_new, seed, pres=None, scores=False):
    """vecs: [B, d] tensor (one row per output); pres: B prefixes of equal length (the reply's first reasoning tokens, read
    unsteered). Returns [(prefix + generated ids, text, finish)]."""
    import torch
    from transformers import GenerationConfig
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
    B = vecs.shape[0]; pres = pres or [[]] * B
    assert len(pres) == B and len({len(q) for q in pres}) == 1
    from transformers import DynamicCache
    # One prefill of the prompt minus its last token at batch 1, copied to every row (a batch of 28 x 5.8k prompts
    # materialises a 27 GiB attention matrix). generate then feeds the last prompt token (+ the prefix) as its first step,
    # which is never edited: a multi-token step is skipped by the hook, and a single-token one by skip = 1.
    cache = DynamicCache(config=model.config.get_text_config())
    model.model.rope_deltas = None   # Qwen3.5 keeps the last call's (batch-sized) rope offsets; recompute them for this prompt
    with torch.no_grad():
        model(input_ids=torch.tensor([ids[:-1]], device="cuda"), past_key_values=cache, use_cache=True)
    cache.reorder_cache(torch.zeros(B, dtype=torch.long, device="cuda"))
    steer.v, steer.skip = vecs[:, None, :].to("cuda"), (0 if pres[0] else 1)
    torch.manual_seed(seed)
    gc = GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=max_new,
                          eos_token_id=eos, pad_token_id=pad)
    x = torch.tensor([ids + list(q) for q in pres], device="cuda")
    with torch.no_grad():
        o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), past_key_values=cache, generation_config=gc,
                           logits_processor=[Sampler(len(ids), body)], return_dict_in_generate=True, output_logits=scores)
    first = o.logits[0].float() if scores else None
    o = o.sequences
    steer.v = None
    del cache
    torch.cuda.empty_cache()
    out = []
    for q, row in zip(pres, o[:, x.shape[1]:].tolist()):
        end = next((i for i, t in enumerate(row) if t in eos), None)
        g = list(q) + (row if end is None else row[:end + 1])
        out.append((g, tok.decode(g, skip_special_tokens=False), "length" if end is None else "stop"))
    return (out, first) if scores else out


def vectors(job, d_model, smoke):
    """arm name -> alpha * gap * d; d by arm dir: probe (E093's u) or randomK (unit, orthogonal to u)."""
    import torch
    if smoke:   # the 0.8B has another width: seeded stand-ins, same code path
        g = np.random.default_rng(0); u = g.standard_normal(d_model); u /= np.linalg.norm(u)
        r0 = g.standard_normal(d_model); r0 -= (r0 @ u) * u; r0 /= np.linalg.norm(r0); gap = job["smoke_gap"]
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(job["artifact_repo"], job["steer"]["path"])
        assert sha(Path(p)) == job["steer"]["sha256"], "steering vectors differ from the config"
        z = np.load(p); u, r0, gap = z["u"], z["rand"], float(z["class_gap"])
        assert u.shape == (d_model,) and abs(gap - 2.463) < 5e-4, (u.shape, gap)
    D = {"probe": u, "random0": r0}
    for k in range(1, 5):
        r = np.random.default_rng(1000 + k).standard_normal(d_model); r -= (r @ u) * u; D[f"random{k}"] = r / np.linalg.norm(r)
    for k, v in D.items():
        assert abs(np.linalg.norm(v) - 1) < 1e-4 and (k == "probe" or abs(float(v @ u)) < 1e-4), k
    return {a["name"]: (a["alpha"] * gap * torch.tensor(D[a["dir"]], dtype=torch.float32)) for a in job["arms"]}, gap


def run(job, smoke=False):
    import torch
    from common import load_generator
    g = job["smoke_generator"] if smoke else job["generator"]
    layer = job["smoke_layer"] if smoke else job["layer"]
    api, name = None, job["out_name"] + ("-smoke" if smoke else "")
    D = HERE / "data" / name; D.mkdir(parents=True, exist_ok=True)
    RES = HERE / "results/e127-cloud" / ("smoke" if smoke else ""); (RES / "stage-logs").mkdir(parents=True, exist_ok=True)
    rows_path = D / "rows.jsonl"
    if not smoke:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        assert api.repo_info(job["artifact_repo"]).private, "artifact repo must be private"
        for rel, expected in job["config_sha256"].items():
            if sha(HERE / rel) != expected:
                raise SystemExit(f"configuration drift since launch: {rel}")
        try:   # resume: the cases an earlier run of the same arms finished
            p = hf_hub_download(job["artifact_repo"], f"{name}/rows.jsonl"); rows_path.write_bytes(Path(p).read_bytes())
        except Exception as e:
            print("no earlier rows:", type(e).__name__, flush=True); rows_path.write_text("")
    else:
        rows_path.write_text("")
    prior = [json.loads(l) for l in open(rows_path)]
    assert all(r["arms_sha"] == arms_sha(job) for r in prior), "earlier rows used other arms: start a new out_name"
    done = {r["case"] for r in prior}

    model, tok, layers = load_generator(g)
    steer = Steer(layers[layer - 1])
    V, gap = vectors(job, model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size, smoke)
    cases, n_mixed = draw(job)
    cases = cases[:job["smoke_cases"] if smoke else job["run_cases"]]
    items = [it for it in items_of(job, tok, cases) if it[0]["case"] not in done]
    K, arms = job["k"], job["arms"]
    assert all(len(it[4]) >= 4 for it in items), "a case has fewer than 4 prefixes at N = 16"
    receipt = dict(gpu=torch.cuda.get_device_name(), started=time.time(), arms=arms, arms_sha=arms_sha(job), gap=gap, layer=layer,
                   pool_cases=n_mixed, skipped_done=len(done), cases=[])
    print(json.dumps({k: receipt[k] for k in ("gpu", "gap", "layer", "pool_cases", "skipped_done")}), "todo", len(items), flush=True)
    last = time.time()

    def flush():
        receipt.update(updated=time.time(), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        (RES / "run.json").write_text(json.dumps(receipt, indent=1))
        if api is not None:
            api.upload_file(path_or_fileobj=str(rows_path), path_in_repo=f"{name}/rows.jsonl", repo_id=job["artifact_repo"])
            api.upload_folder(repo_id=job["artifact_repo"], folder_path=RES, path_in_repo=f"{name}/results")

    max_new = job["smoke_max_tokens"] if smoke else None
    for ci, (m, body, ids, rate, pre) in enumerate(items):
        t0, e0 = time.time(), steer.edits
        jobs = [(a, p) for a in arms for p in pre[:a.get("k", K)]]
        gens = []
        rpc = job["smoke_rows_per_call"] if smoke else job["rows_per_call"]
        for s in range(0, len(jobs), rpc):
            chunk = jobs[s:s + rpc]
            vec = torch.stack([V[a["name"]] for a, _ in chunk])
            gens += generate(model, tok, steer, ids, body, vec, (max_new or body["max_tokens"]) - job["n"], case_seed(m["case"]) + s,
                             [p[1] for _, p in chunk])
        rows = []
        for (a, p), (gid, text, fin) in zip(jobs, gens):
            row, _ = rows_of("realistic", "test", m, p[0], case_seed(m["case"]), gid, text, fin, tok, [8, 16])
            row.update(arm=a["name"], dir=a["dir"], alpha=a["alpha"], arms_sha=arms_sha(job), dataset_rate=rate,
                       dataset_strict_this_rollout=p[2])
            rows.append(row)
        with open(rows_path, "a") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        s = dict(case=m["case"], seconds=round(time.time() - t0, 1), out_tokens=sum(x["tokens"] for x in rows), edits=steer.edits - e0,
                 strict={a["name"]: sum(x["violates_strict"] for x in rows if x["arm"] == a["name"]) for a in arms})
        receipt["cases"].append(s); print(ci, json.dumps(s), flush=True)
        if smoke:
            assert s["edits"] > 0, "the steering hook never fired"
        if time.time() - last > job.get("upload_every_s", 600):
            flush(); last = time.time()
    receipt.update(finished=time.time(), complete=True); flush()
    steer.h.remove()
    if smoke:
        smoke_checks(job, model, tok, steer, layers, items[0], V, max_new)


def smoke_checks(job, model, tok, steer, layers, item, V, max_new):
    """The copied cache + prefix step equals one full forward; the hook is a no-op at alpha = 0 and changes the text at a large alpha."""
    import torch
    m, body, ids, _, pre = item
    q = [p[1] for p in pre[:2]]
    off = type("Off", (), {"v": None, "skip": 0})()
    z = torch.zeros(2, next(iter(V.values())).shape[0])
    b, first = generate(model, tok, off, ids, body, z, max_new, 1, q, scores=True)
    with torch.no_grad():
        full = model(input_ids=torch.tensor([ids + q[0]], device="cuda")).logits[0, -1].float()
    cos = float(torch.nn.functional.cosine_similarity(first[0], full, 0))
    assert cos > 0.99 and int(first[0].argmax()) == int(full.argmax()), ("cached prefix step differs from a full forward", cos)
    steer = Steer(layers[job["smoke_layer"] - 1])
    a = generate(model, tok, steer, ids, body, z, max_new, 1, q)
    steer.h.remove()
    div = [next((i for i, (s, t) in enumerate(zip(x[0], y[0])) if s != t), None) for x, y in zip(a, b)]
    assert div == [None] * len(a), ("alpha = 0 with the hook differs from no hook at token", div)
    assert all(x[0][:len(p)] == p for x, p in zip(a, q)), "the reply does not start with its prefix"
    steer = Steer(layers[job["smoke_layer"] - 1])
    big = torch.stack([V[max(job["arms"], key=lambda x: abs(x["alpha"]))["name"]] * 4] * 2)
    c = generate(model, tok, steer, ids, body, big, max_new, 1, q)
    steer.h.remove()
    assert [x[0] for x in c] != [x[0] for x in a], "a large edit left the text unchanged"
    print("smoke checks pass: cache cosine", round(cos, 5), "| hook no-op at 0 | text moves at 4x the largest arm", flush=True)
    print("sample (alpha 0):", a[0][1][:300].replace("\n", " "), flush=True)
    print("sample (large):  ", c[0][1][:300].replace("\n", " "), flush=True)


def prep(job):
    from transformers import AutoTokenizer
    g = job["generator"]; tok = AutoTokenizer.from_pretrained(g["model_id"], revision=g["revision"])
    cases, n_pool = draw(job)
    items = items_of(job, tok, cases)
    P = sorted(len(i) for _, _, i, _, _ in items); npre = [len(x[4]) for x in items[:job["run_cases"]]]
    rows = sum(min(a.get("k", job["k"]), n) for n in npre for a in job["arms"])
    print(dict(pool_cases=n_pool, drawn=len(cases), run_cases=job["run_cases"], arms=len(job["arms"]), replies=rows,
               prefixes_per_case_min_mean=[min(npre), round(float(np.mean(npre)), 2)], prompt_tokens_p50_max=[P[len(P) // 2], P[-1]],
               dataset_eval_rate=round(float(np.mean([r for _, r in cases])), 3),
               dataset_strict_on_prefix_rollouts=round(float(np.mean([np.mean([p[2] for p in x[4]]) for x in items[:job["run_cases"]]])), 3),
               max_tokens={b["max_tokens"] for _, b, _, _, _ in items}, sampler={(b["temperature"], b["top_p"], b["top_k"], b["presence_penalty"]) for _, b, _, _, _ in items}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("stage", choices=["prep", "smoke", "run"])
    a = ap.parse_args(); job = yaml.safe_load((HERE / a.config).read_text()); os.chdir(HERE)
    {"prep": lambda: prep(job), "smoke": lambda: run(job, smoke=True), "run": lambda: run(job)}[a.stage]()
