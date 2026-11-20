"""E125 activation states for gpt-oss-20b (experiments/E125-agentdojo-gptoss-heldout.md). Pod, HF transformers, after the vLLM server stops.

  python e125_states.py smoke                 # load, forward one rendered conversation, check the next token and time it
  python e125_states.py run --data DIR        # -> DIR/states/<split>-n0.npy (+ -meta.jsonl), <split>-n16.npy (+ -meta.jsonl), receipt.json

N = 0: residual after the configured blocks at the checkpoint prompt's last token (`assistant` of `<|start|>assistant`), one per input with a
checkpoint, from the prefix record's token ids (the ids the vLLM server was sent). N = 16: the same prompt plus U's checkpoint-call tokens
`<|channel|>analysis<|message|>` + 16 analysis tokens (vLLM's sampled ids), read at the 16th analysis token, one per U run (T applicable).
The prompt is prefilled in chunks into a KV cache; each run's 19 tokens are forwarded on a copy of it.
Checks recorded in the receipt: the model's top next token after each checkpoint prompt (expected `<|channel|>`), and the mean
log-probability HF assigns to the 16 sampled analysis tokens (a rendering or position mismatch shows up as very low values).
"""
import argparse, copy, json, sys, time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "configs/e125.yaml").read_text())
M = yaml.safe_load((HERE / "configs/e117.yaml").read_text())["models"]["gpt-oss-20b"]
LAYERS = CFG["layers"]; N16 = 16


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from transformers import Mxfp4Config
        q = dict(quantization_config=Mxfp4Config(dequantize=True))
    except ImportError:
        q = {}
    model = AutoModelForCausalLM.from_pretrained(M["hf_id"], revision=M["revision"], dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager", **q).eval()
    tok = AutoTokenizer.from_pretrained(M["hf_id"], revision=M["revision"])
    assert model.config.num_hidden_layers == 24 and max(LAYERS) < 24
    return model, tok


@torch.no_grad()
def prefill(model, ids):
    cache, x, ch, out = None, torch.tensor([ids], device="cuda"), CFG["prefill_chunk"], None
    for s in range(0, len(ids), ch):
        last = s + ch >= len(ids)
        out = model(x[:, s:s + ch], past_key_values=cache, use_cache=True, output_hidden_states=last, logits_to_keep=1); cache = out.past_key_values
    h = torch.stack([out.hidden_states[l][0, -1] for l in LAYERS]).float().cpu().numpy()
    return cache, h, out.logits[0, -1].float()


@torch.no_grad()
def extend(model, cache, last_logits, ext):
    c = copy.deepcopy(cache)
    out = model(torch.tensor([ext], device="cuda"), past_key_values=c, use_cache=True, output_hidden_states=True)
    h = torch.stack([out.hidden_states[l][0, -1] for l in LAYERS]).float().cpu().numpy()
    lg = torch.cat([last_logits[None], out.logits[0, :-1].float()]).log_softmax(-1)
    lp = lg[torch.arange(len(ext)), torch.tensor(ext, device=lg.device)].cpu().numpy()
    return h, float(lp[3:].mean())   # the 16 analysis tokens, after the 3 header tokens


def smoke():
    import openai_harmony as H
    enc = H.load_harmony_encoding(H.HarmonyEncodingName.HARMONY_GPT_OSS)
    conv = H.Conversation.from_messages([H.Message.from_role_and_content(H.Role.SYSTEM, H.SystemContent.new()),
                                         H.Message.from_role_and_content(H.Role.USER, "Summarise this: " + "The quarterly report shows growth. " * 400)])
    ids = enc.render_conversation_for_completion(conv, H.Role.ASSISTANT)
    t0 = time.time(); model, tok = load(); t_load = time.time() - t0
    t0 = time.time(); cache, h, lg = prefill(model, ids); torch.cuda.synchronize(); t_fwd = time.time() - t0
    top = tok.convert_ids_to_tokens(int(lg.argmax()))
    ext = enc.encode("<|channel|>analysis<|message|>", allowed_special="all") + tok("The user wants a short summary of a repeated sentence about growth in the report.", add_special_tokens=False).input_ids[:N16]
    h16, lp = extend(model, cache, lg, ext)
    print(json.dumps(dict(prompt_tokens=len(ids), load_s=round(t_load, 1), prefill_s=round(t_fwd, 2), top_next=top, n16_mean_logprob=round(lp, 2),
                          state_shape=list(h.shape), finite=bool(np.isfinite(h).all() and np.isfinite(h16).all()),
                          peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))), flush=True)
    assert top == "<|channel|>", f"next token after <|start|>assistant is {top}"
    assert np.isfinite(h).all() and np.isfinite(h16).all()


def run(data):
    import openai_harmony as H
    enc = H.load_harmony_encoding(H.HarmonyEncodingName.HARMONY_GPT_OSS); hd = enc.encode("<|channel|>analysis<|message|>", allowed_special="all")
    end = enc.encode("<|end|>", allowed_special="all")[0]
    model, tok = load(); out = Path(data) / "states"; out.mkdir(parents=True, exist_ok=True); receipt = dict(splits={})
    for split in ("dev", "test"):
        P = [r for r in map(json.loads, open(Path(data) / f"prefix-{split}.jsonl")) if r["checkpoint"]]
        U = {}
        for r in map(json.loads, open(Path(data) / f"cont-{split}-U.jsonl")):
            U.setdefault(r["input"], []).append(r)
        S0, M0, S16, M16, tops, lps = [], [], [], [], [], []; t0 = time.time()
        for n, p in enumerate(P):
            ids = p["checkpoint"]["ids"]; cache, h, lg = prefill(model, ids)
            S0.append(h.astype(np.float16)); M0.append(dict(input=p["input"], prompt_tokens=len(ids)))
            tops.append(tok.convert_ids_to_tokens(int(lg.argmax())))
            for u in sorted(U.get(p["input"], []), key=lambda r: r["r"]):
                ki = (u.get("k_ids") or {}).get("ids") or []
                if ki[:3] != hd or len(ki) < 3 + N16 or end in ki[3:3 + N16]:
                    continue
                h16, lp = extend(model, cache, lg, ki[:3 + N16])
                S16.append(h16.astype(np.float16)); M16.append(dict(input=p["input"], r=u["r"], ids_source=u["k_ids"]["source"], mean_logprob=round(lp, 3)))
                lps.append(lp)
            del cache; torch.cuda.empty_cache()
            if n % 20 == 0:
                print(f"[e125 states] {split} {n + 1}/{len(P)} n16={len(S16)} {time.time() - t0:.0f}s", flush=True)
        np.save(out / f"{split}-n0.npy", np.stack(S0)); (out / f"{split}-n0-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M0))
        if S16:
            np.save(out / f"{split}-n16.npy", np.stack(S16))
        (out / f"{split}-n16-meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in M16))
        receipt["splits"][split] = dict(checkpoints=len(P), n16=len(S16), top_next_channel=sum(t == "<|channel|>" for t in tops) / max(1, len(tops)),
                                        n16_mean_logprob=dict(mean=float(np.mean(lps)) if lps else None, p05=float(np.percentile(lps, 5)) if lps else None),
                                        seconds=round(time.time() - t0, 1))
        print(json.dumps({split: receipt["splits"][split]}), flush=True)
    receipt.update(layers=LAYERS, gpu=torch.cuda.get_device_name(), peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["smoke", "run"]); ap.add_argument("--data", default=str(HERE / "data/e125"))
    a = ap.parse_args()
    smoke() if a.cmd == "smoke" else run(a.data)
