"""E129 activation states (experiments/E129-agentdojo-multi-attack.md). Pod, HF transformers, after the vLLM server stops.

  python e129_states.py MODEL smoke [--small]            # load, forward one rendered conversation, check finiteness / next token, time it
  python e129_states.py MODEL run --data DIR --split S   # -> DIR/states/<S>-n0.npy (+ -meta.jsonl), <S>-n16.npy (+ -meta.jsonl), receipt-<S>.json

N = 0: the state at the checkpoint prompt's last token, one per input with a checkpoint, from the prefix record's token ids (the ids the
vLLM server was sent). N = 16: the same prompt plus U's first 16 reasoning tokens (vLLM's sampled ids; for gpt-oss after the 3 Harmony
header tokens), read at the 16th, one per U run with T applicable. The prompt is prefilled in chunks into a KV cache; each run's
extension is forwarded on a copy of it.
- gpt-oss-20b: E125's code and layers (e125_states: HF hidden_states after blocks 6 9 12 15 18 20 21 22 23 of 24).
- qwen35-9b: forward-hook outputs of decoder layers 8 12 16 20 24 26 28 30 32 (common.LayerTap: E079 / E115's stored quantity, so E115's
  frozen InjecForks heads read the same thing); the prompt ends `<think>\\n` (asserted).
Receipt checks: gpt-oss top next token `<|channel|>`; both: mean log-probability of the 16 sampled tokens (a rendering or position mismatch
shows up as very low values).
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CFG = yaml.safe_load((HERE / "configs/e129.yaml").read_text())
N16 = 16


class Qwen:
    def __init__(self, small=False):
        from common import LayerTap, load_generator
        m = CFG["models"]["qwen35-9b"]; g = m["smoke_generator"] if small else m["generator"]
        self.layers_id = m["smoke_layers"] if small else m["layers"]
        self.model, self.tok, layers = load_generator(g, "cuda"); self.lm = self.model
        self.grab = {}; self.tap = LayerTap(layers, self.layers_id, lambda li, h: self.grab.__setitem__(li, h[0, -1].float().cpu().numpy()))

    @torch.no_grad()
    def prefill(self, ids):
        cache, x, ch, out = None, torch.tensor([ids], device="cuda"), CFG["prefill_chunk"], None
        for s in range(0, len(ids), ch):
            out = self.model(input_ids=x[:, s:s + ch], past_key_values=cache, use_cache=True); cache = out.past_key_values
        h = np.stack([self.grab[li] for li in self.layers_id])
        return cache, h, out.logits[0, -1].float()

    @torch.no_grad()
    def extend(self, cache, last_logits, ext):
        c = copy.deepcopy(cache)
        out = self.model(input_ids=torch.tensor([ext], device="cuda"), past_key_values=c, use_cache=True)
        h = np.stack([self.grab[li] for li in self.layers_id])
        lg = torch.cat([last_logits[None], out.logits[0, :-1].float()]).log_softmax(-1)
        lp = lg[torch.arange(len(ext)), torch.tensor(ext, device=lg.device)].cpu().numpy()
        return h, float(lp.mean())


class GptOss:
    def __init__(self):
        import e125_states as S
        assert S.LAYERS == CFG["models"]["gpt-oss-20b"]["layers"]
        self.S = S; self.model, self.tok = S.load()

    def prefill(self, ids):
        return self.S.prefill(self.model, ids)

    def extend(self, cache, lg, ext):
        return self.S.extend(self.model, cache, lg, ext)


def smoke(model, small):
    if model == "gpt-oss-20b":
        import e125_states as S
        return S.smoke()
    t0 = time.time(); M = Qwen(small); t_load = time.time() - t0
    msgs = [{"role": "user", "content": "Summarise this: " + "The quarterly report shows growth. " * 400}]
    chat = M.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False); assert chat.endswith("<think>\n"), chat[-30:]
    ids = M.tok(chat, add_special_tokens=False).input_ids
    t0 = time.time(); cache, h, lg = M.prefill(ids); torch.cuda.synchronize(); t_fwd = time.time() - t0
    ext = M.tok("The user wants a short summary of a repeated sentence about growth in the quarterly report, so I will", add_special_tokens=False).input_ids[:N16]
    h16, lp = M.extend(cache, lg, ext)
    h16b, lpb = M.extend(cache, lg, ext)   # the cache copy leaves the prefilled cache untouched: a second extension is identical
    full = M.tok(chat, add_special_tokens=False).input_ids + ext
    with torch.no_grad():
        M.model(input_ids=torch.tensor([full], device="cuda"))
    hf = np.stack([M.grab[li] for li in M.layers_id])
    cos = float(np.min([np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b) for a, b in zip(h16, hf)]))
    print(json.dumps(dict(prompt_tokens=len(ids), load_s=round(t_load, 1), prefill_s=round(t_fwd, 2), n16_mean_logprob=round(lp, 2),
                          repeat_identical=bool(np.allclose(h16, h16b, atol=1e-3)), cached_vs_full_min_cos=round(cos, 5), state_shape=list(h.shape),
                          finite=bool(np.isfinite(h).all() and np.isfinite(h16).all()), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)
    assert np.isfinite(h).all() and np.isfinite(h16).all() and np.allclose(h16, h16b, atol=1e-3) and cos > 0.99


def run(model, data, split):
    M = GptOss() if model == "gpt-oss-20b" else Qwen()
    out = Path(data) / "states"; out.mkdir(parents=True, exist_ok=True)
    if model == "gpt-oss-20b":
        import openai_harmony as H
        enc = H.load_harmony_encoding(H.HarmonyEncodingName.HARMONY_GPT_OSS); hd = enc.encode("<|channel|>analysis<|message|>", allowed_special="all")
        end = enc.encode("<|end|>", allowed_special="all")[0]
        ext_of = lambda ki: ki[:3 + N16] if ki[:3] == hd and len(ki) >= 3 + N16 and end not in ki[3:3 + N16] else None
        name = lambda i: M.tok.convert_ids_to_tokens(i)
    else:
        close = M.tok.convert_tokens_to_ids("</think>"); stops = {M.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
        ext_of = lambda ki: ki[:N16] if len(ki) >= N16 and close not in ki[:N16] and not stops & set(ki[:N16]) else None
        name = lambda i: M.tok.convert_ids_to_tokens(i)
    P = [r for r in map(json.loads, open(Path(data) / f"prefix-{split}.jsonl")) if r["checkpoint"]]
    U = {}
    for r in map(json.loads, open(Path(data) / f"cont-{split}-U.jsonl")):
        U.setdefault(r["input"], []).append(r)
    S0, M0, S16, M16, tops, lps = [], [], [], [], [], []; t0 = time.time()
    for n, p in enumerate(P):
        ids = p["checkpoint"]["ids"]
        if model == "qwen35-9b":
            assert M.tok.decode(ids[-3:]).endswith("<think>\n"), "qwen checkpoint prompt must end <think>\\n"
        cache, h, lg = M.prefill(ids)
        S0.append(h.astype(np.float16)); M0.append(dict(input=p["input"], prompt_tokens=len(ids)))
        tops.append(name(int(lg.argmax())))
        for u in sorted(U.get(p["input"], []), key=lambda r: r["r"]):
            ext = ext_of((u.get("k_ids") or {}).get("ids") or [])
            if ext is None:
                continue
            h16, lp = M.extend(cache, lg, ext)
            S16.append(h16.astype(np.float16)); M16.append(dict(input=p["input"], r=u["r"], ids_source=u["k_ids"]["source"], mean_logprob=round(lp, 3)))
            lps.append(lp)
        del cache; torch.cuda.empty_cache()
        if n % 20 == 0:
            print(f"[e129 states] {model} {split} {n + 1}/{len(P)} n16={len(S16)} {time.time() - t0:.0f}s", flush=True)
    np.save(out / f"{split}-n0.npy", np.stack(S0)); (out / f"{split}-n0-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M0))
    if S16:
        np.save(out / f"{split}-n16.npy", np.stack(S16))
    (out / f"{split}-n16-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M16))
    rc = dict(model=model, split=split, checkpoints=len(P), n16=len(S16), layers=CFG["models"][model]["layers"],
              top_next_channel=sum(t == "<|channel|>" for t in tops) / max(1, len(tops)) if model == "gpt-oss-20b" else None,
              top_next_examples=sorted(set(tops))[:20], n16_mean_logprob=dict(mean=float(np.mean(lps)) if lps else None, p05=float(np.percentile(lps, 5)) if lps else None),
              seconds=round(time.time() - t0, 1), gpu=torch.cuda.get_device_name(), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    (out / f"receipt-{split}.json").write_text(json.dumps(rc, indent=1)); print(json.dumps(rc), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model", choices=list(CFG["models"])); ap.add_argument("cmd", choices=["smoke", "run"])
    ap.add_argument("--data"); ap.add_argument("--split", choices=["dev", "test"]); ap.add_argument("--small", action="store_true")
    a = ap.parse_args()
    smoke(a.model, a.small) if a.cmd == "smoke" else run(a.model, a.data, a.split)
