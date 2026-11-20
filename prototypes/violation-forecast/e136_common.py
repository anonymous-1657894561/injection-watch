"""E136 shared pieces: a fine-tuned DeBERTa-v3-base classifier on the visible reasoning prefix (experiments/E136-*.md).

Text = r["reasoning"][:r["cut"][N]] exactly as E134 (e134_common.pre; nothing from the prompt), label `violates` (v1.2).
Fit on every InjecForks train row with a prefix at N, per config and N. Score = logit(violates) - logit(not).
Memory (RTX 2060, shared): the 128k x 768 word-embedding matrix is frozen (its Adam state alone is ~1.2 GB); every other weight
(embedding LayerNorm, the 12 layers, the head) trains. fp16 autocast + GradScaler, AdamW (wd 0.01, none on bias / LayerNorm),
linear warm-up over the first 10 % of an EMAX = 4-epoch schedule then linear decay; `stop` ends training after that many epochs
of the same 4-epoch schedule, so a model stopped at epoch e is the grid's epoch-e checkpoint. Batch 32, grad-norm clip 1.0.
Models are saved fp16 under E136_MODELS (default data/e136-models, git-ignored).
"""
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e134_common import load, pre, read_scores  # noqa: E402,F401

HERE = Path(__file__).resolve().parent
SC = HERE / "results/injecforks/scores"
OUT = HERE / "results/e136"
MODELS = Path(os.environ.get("E136_MODELS", HERE / "data/e136-models"))
MODEL = "microsoft/deberta-v3-base"
NS = [8, 16, 32, 64, 128]
MAXLEN, BS, EMAX, WARM, WD = 256, 32, 4, 0.1, 0.01
MICRO = int(os.environ.get("E136_MICRO", BS))  # gradient-accumulation chunk; MICRO = BS is the E136 run
_tok = None
GPU_LOCK = os.environ.get("GPU_LOCK", os.path.join(tempfile.gettempdir(), "gpu.lock"))


class gpu_lock:
    """flock on the shared GPU lock file around each GPU job (the GPU is shared with other agents; one job at a time)."""
    def __enter__(self):
        import fcntl
        self.f = open(GPU_LOCK, "a"); fcntl.flock(self.f, fcntl.LOCK_EX); return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(MODEL)
    return _tok


def encode(texts):
    ids = tok()(list(texts), truncation=False)["input_ids"]
    assert max(map(len, ids)) <= MAXLEN, max(map(len, ids))
    return ids


def _pad(seqs):
    import torch
    L = max(map(len, seqs)); pid = tok().pad_token_id
    x = torch.full((len(seqs), L), pid, dtype=torch.long); m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return x.cuda(), m.cuda()


def predict(model, texts, bs=128):
    """logit(1) - logit(0) per text, in input order."""
    import torch
    ids = encode(texts); order = np.argsort([len(s) for s in ids], kind="stable"); out = np.empty(len(ids))
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(ids), bs):
            j = order[i:i + bs]; x, m = _pad([ids[k] for k in j])
            lg = model(input_ids=x, attention_mask=m).logits.float()
            out[j] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    model.train()
    return out


def new_model(seed):
    import torch
    from transformers import AutoModelForSequenceClassification
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, dtype=torch.float32).cuda()
    m.deberta.embeddings.word_embeddings.weight.requires_grad_(False)
    return m


def train_rows(cfg, N):
    return [r for r in load(cfg, "train", subset=None) if r["cut"].get(N) is not None]


def train(cfg, N, lr, cw, seed, stop=EMAX, evals=None, save=None, log=print):
    """Fine-tune on train (cfg, N). evals: {name: rows}; scored after every epoch. Returns {epoch: {name: {row_id: score}}}."""
    import torch
    from transformers import get_linear_schedule_with_warmup
    assert MICRO >= BS or cw != "balanced", "chunked loss is only exact without class weights"
    tr = train_rows(cfg, N)
    ids = encode([pre(r, N) for r in tr]); y = np.array([int(bool(r["violates"])) for r in tr])
    model = new_model(seed)
    dec = [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias", "LayerNorm"))]
    nod = [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in ("bias", "LayerNorm"))]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=WD), dict(params=nod, weight_decay=0.0)], lr=lr)
    spe = int(np.ceil(len(ids) / BS)); sched = get_linear_schedule_with_warmup(opt, int(WARM * EMAX * spe), EMAX * spe)
    scaler = torch.amp.GradScaler("cuda")
    w = torch.tensor([len(y) / (2 * (y == 0).sum()), len(y) / (2 * (y == 1).sum())], dtype=torch.float32).cuda() if cw == "balanced" else None
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(seed); yt = torch.tensor(y).cuda(); out = {}
    ev = {k: [r for r in v if r["cut"].get(N) is not None] for k, v in (evals or {}).items()}
    model.train()
    for ep in range(1, stop + 1):
        perm = rng.permutation(len(ids)); tot = 0.0
        for i in range(0, len(ids), BS):
            j = perm[i:i + BS]; opt.zero_grad(set_to_none=True); loss = 0.0
            for c in range(0, len(j), MICRO):  # one chunk when MICRO >= BS (E136); exact mean loss when cw is None
                jc = j[c:c + MICRO]; x, m = _pad([ids[k] for k in jc])
                with torch.autocast("cuda", dtype=torch.float16):
                    lg = model(input_ids=x, attention_mask=m).logits
                lc = lossf(lg.float(), yt[jc]) * (len(jc) / len(j))
                scaler.scale(lc).backward(); loss += float(lc)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss) * len(j)
        out[ep] = {k: dict(zip([r["row_id"] for r in rs], map(float, predict(model, [pre(r, N) for r in rs])))) for k, rs in ev.items()}
        log(f"{cfg} N={N} lr={lr} cw={cw} seed={seed} epoch {ep}: train loss {tot / len(ids):.4f}")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in model.state_dict().items()}, save)
    del model, opt; torch.cuda.empty_cache()
    return out


def model_path(cfg, N, seed):
    return MODELS / f"{cfg}-N{N}-s{seed}.pt"


def load_model(cfg, N, seed):
    import torch
    m = new_model(0)
    m.load_state_dict({k: v.float() for k, v in torch.load(model_path(cfg, N, seed), map_location="cuda").items()})
    m.eval()
    return m


def score_ens(cfg, N, seeds, rows):
    """Mean over the seeds' saved models of logit(1) - logit(0) on the rows with a prefix at N; {row_id: (ens, [per seed])}."""
    import torch
    rs = [r for r in rows if r["cut"].get(N) is not None]
    if not rs:
        return {}
    T = [pre(r, N) for r in rs]; per = []
    for s in seeds:
        m = load_model(cfg, N, s); per.append(predict(m, T)); del m; torch.cuda.empty_cache()
    P = np.stack(per, 1)
    return {r["row_id"]: (float(P[i].mean()), [float(v) for v in P[i]]) for i, r in enumerate(rs)}
