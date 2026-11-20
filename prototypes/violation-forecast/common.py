"""Shared pieces: config, model loading (nf4), prompt rendering, layer hooks, exact-match labels.

Layer convention, identical for every generator: `layers` in the config are 1-indexed decoder
layers, and the state read is that layer's OUTPUT residual (before the final norm), captured by a
forward hook. So layer 32 of a 32-layer model is the last block's output, not the normed
last_hidden_state.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
REFUSAL_RE = re.compile(
    r"\b(can(?:'|no)t|cannot|unable to|not able to|won'?t|sorry|not (?:allowed|permitted|authori[sz]ed)"
    r"|must (?:decline|refuse)|I (?:do not|don't) (?:share|disclose|reveal)|decline)\b", re.I)


def load_cfg(path):
    return yaml.safe_load(open(path))


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


def data_dir(cfg):
    return HERE / cfg.get("data_dir", "data")


def scenarios(path=None):
    return [json.loads(l) for l in open(path or HERE / "data/scenarios.jsonl")]


def loader_trusts(gcfg):
    return gcfg["loader"] == "dqwen_remote"


def load_generator(gcfg, device=None):
    device = device or os.environ.get("E15_DEVICE", "cuda")
    """Returns (model, tokenizer, decoder_layers). nf4 weights (default) or bf16 (`quant: bf16`), pinned revision."""
    from transformers import AutoTokenizer
    kw = dict(revision=gcfg["revision"], dtype=torch.bfloat16, device_map={"": device})
    if gcfg.get("quant", "nf4") == "nf4":        # the local 6 GB default; `quant: bf16` on a pod
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(gcfg.get("tokenizer_id", gcfg["model_id"]),
                                        revision=gcfg.get("tokenizer_revision", gcfg["revision"]),
                                        trust_remote_code=loader_trusts(gcfg))
    loader = gcfg["loader"]
    if loader == "dqwen_remote":
        from transformers import AutoModel
        model = AutoModel.from_pretrained(gcfg["model_id"], trust_remote_code=True, **kw)
        model._dg_tok = tok  # the sampler's own tokenizer cache; same repo and revision
        layers = model.model.layers
    elif loader == "qwen35_vlm_text":
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(gcfg["model_id"], **kw)
        layers = model.model.language_model.layers
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(gcfg["model_id"], **kw)
        layers = model.model.layers
    model.eval()
    assert len(layers) == gcfg["n_layers"], (len(layers), gcfg["n_layers"])
    return model, tok, layers


def render(tok, gcfg, system, user):
    if gcfg["prompt_format"] == "chat":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return tok.apply_chat_template(
            msgs,
            add_generation_prompt=True, tokenize=False, **gcfg.get("chat_kwargs", {}))
    if gcfg["prompt_format"] == "transcript":
        return (f"{system}\n\n" if system else "") + f"User: {user}\nAssistant:"
    raise ValueError(gcfg["prompt_format"])


class LayerTap:
    """Forward hooks on the chosen decoder layers. Read-only: returns nothing, so the
    generator's computation is untouched. `on_output(layer_idx, hidden)` gets [B, L, d]."""

    def __init__(self, layers, which, on_output):
        self.handles = []
        for li in which:
            mod = layers[li - 1]
            self.handles.append(mod.register_forward_hook(self._mk(li, on_output)))

    @staticmethod
    def _mk(li, fn):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            fn(li, h.detach())
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def pieces(tok, ids):
    """Per-token decoded strings; joined they are the text the labels are computed on."""
    return [tok.decode([int(t)], skip_special_tokens=False) for t in ids]


def answer_end(text, stops):
    """Char offset where the answer ends: first stop string, else the full text."""
    hits = [text.find(s) for s in stops if s and text.find(s) >= 0]
    return min(hits) if hits else len(text)


def refusal(text):
    return bool(REFUSAL_RE.search(text))


def _sync():
    if torch.cuda.is_available() and os.environ.get("E15_DEVICE", "cuda") == "cuda":
        torch.cuda.synchronize()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _C:
            def __enter__(s):
                _sync()
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                _sync()
                timer.t.setdefault(key, []).append(time.perf_counter() - s.t0)
        return _C()
