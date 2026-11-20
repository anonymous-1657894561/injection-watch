"""How the E101 observer (DeepSeek-R1-Distill-Llama-8B) reads an InjecForks conversation and the 9B's reasoning prefix.
Pure text and tokenizer work (CPU); injecforks_observer_states.py forwards what this builds. Decisions: docs/plan-2026-10-30-distill-observer.md.

The conversation is the exact OpenRouter request body the 9B answered (prompts/<config>/<split>.jsonl), rendered with the
observer's OWN pinned chat template. That template (tokenizer_config.json at the pinned revision) has no `tools` slot and no
reasoning slot for an earlier assistant turn, renders a call as DeepSeek-V3 markup, and after a tool result emits NO
`<｜Assistant｜>` header: the assistant's continuation follows `<｜tool▁outputs▁end｜>` directly. So:

  system     the request's system text, then "## Tools" and one JSON line per tool (the same JSON lines the 9B's <tools> held)
  user       verbatim
  assistant  the call through the template's own tool_calls branch (name + the request's argument JSON string, verbatim);
             the 9B's step-1 reasoning is dropped, as the template drops it (content None -> no reasoning rendered).
             One patch (CALL_CLOSE): the template never closes a single call; DeepSeek-R1-0528's template (TEMPLATE_FIX_SOURCE)
             emits CALL_CLOSE after the calls loop, so it is inserted there. test_observer_render.py asserts the patched text
             equals R1-0528's own rendering of the same messages.
  tool       the result string verbatim, through the template's tool-output branch
  then       <think>\\n + PREFIX: the prefix is the current assistant turn's reasoning

PREFIX = reasoning[:cut[N]] (cuts/ character offsets: the 9B's first N tokens), tokenized alone, as the 9B pipeline
tokenized its reasoning. The read is its last token. N = 0 reads the context's last token (`\\n` after <think>).
The DeepSeek tool markup is plain text to this tokenizer (11 BPE tokens each, not special tokens); <think> is one added token.
"""
import json

TOOLS_HEADER = "\n\n## Tools\n\nYou have access to the following functions:\n\n"
OUT_END, THINK = "<｜tool▁outputs▁end｜>", "<think>"
CALL_END, OUT_BEGIN = "<｜tool▁call▁end｜>", "<｜tool▁outputs▁begin｜>"
CALLS_END = "<｜tool▁calls▁end｜>"
CALL_CLOSE = CALLS_END + "<｜end▁of▁sentence｜>"
TEMPLATE_FIX_SOURCE = ("deepseek-ai/DeepSeek-R1-0528", "4236a6af538feda4548eca9ab308586007567f52")


def tool_lines(body):
    """One JSON line per tool, as the Qwen3.5 template's `tool | tojson` wrote them inside <tools> (asserted per case)."""
    return [json.dumps(t, ensure_ascii=False) for t in body["tools"]]


def observer_messages(body):
    """The request body's four messages, in the shape the R1-Distill template renders."""
    m = body["messages"]
    assert [x["role"] for x in m] == ["system", "user", "assistant", "tool"], [x["role"] for x in m]
    a = m[2]
    assert a["content"] is None and len(a["tool_calls"]) == 1, "one call, no content: the template's tool_calls branch"
    c = a["tool_calls"][0]
    assert isinstance(c["function"]["arguments"], str) and c["type"] == "function"
    json.loads(c["function"]["arguments"])
    return [{"role": "system", "content": m[0]["content"] + TOOLS_HEADER + "\n".join(tool_lines(body))},
            {"role": "user", "content": m[1]["content"]},
            {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": dict(c["function"])}]},
            {"role": "tool", "content": m[3]["content"]}]


def conversation(tok, body):
    """The pinned template's rendering, with the single call closed as R1-0528 closes it."""
    s = tok.apply_chat_template(observer_messages(body), add_generation_prompt=True, tokenize=False)
    assert s.endswith(OUT_END), s[-80:]   # tool last: the template adds no generation prompt, the turn continues here
    assert s.count(CALL_END + OUT_BEGIN) == 1 and CALLS_END not in s, "template no longer leaves the call open"
    return s.replace(CALL_END + OUT_BEGIN, CALL_END + CALL_CLOSE + OUT_BEGIN)


def context(tok, body):
    """The observer's text up to (not including) the prefix."""
    return conversation(tok, body) + THINK + "\n"   # what the template's generation prompt opens the reasoning with


def check_context(tok, body, ctx):
    """Every content piece verbatim; the result once; the call closed once; <think> opened exactly once, at the end."""
    m = body["messages"]; c = m[2]["tool_calls"][0]["function"]
    assert ctx.startswith(tok.bos_token + m[0]["content"]), "system text must follow BOS directly"
    for piece in [m[0]["content"], m[1]["content"], m[3]["content"], c["arguments"]] + tool_lines(body):
        assert ctx.count(piece) >= 1, ("content missing", piece[:60])
    assert ctx.count(m[3]["content"]) == 1, "tool result rendered more than once"
    assert ctx.count("<｜tool▁output▁begin｜>") == 1 and ctx.count(tok.bos_token) == 1
    assert ctx.count(CALL_CLOSE) == 1 and ctx.count(CALL_END + CALL_CLOSE + OUT_BEGIN) == 1
    assert ctx.count(THINK) == 1 and "</think>" not in ctx and ctx.endswith(OUT_END + THINK + "\n")
    r1 = m[2].get("reasoning") or ""
    if r1 and r1 not in m[0]["content"] + m[1]["content"] + m[3]["content"]:
        assert r1 not in ctx, "the 9B's step-1 reasoning reached the observer context (the template drops it)"


def ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def context_ids(tok, body):
    """(context text, ids), with the ids checked against the text."""
    ctx = context(tok, body); check_context(tok, body, ctx)
    x = ids(tok, ctx)
    assert x[0] == tok.bos_token_id and x.count(tok.bos_token_id) == 1
    assert tok.decode(x) == ctx, "context does not round-trip"
    think = tok.convert_tokens_to_ids(THINK)
    assert x[-2] == think and tok.decode(x[-1:]) == "\n" and x.count(think) == 1
    return ctx, x


def prefix_chains(tok, reasoning, cut, ncut):
    """Tokenize reasoning[:cut[N]] alone for each N present, and group N whose ids nest (one forward serves them).
    Returns ([chain ids], {N: (chain index, position of the last prefix token in that chain)})."""
    chains, reads = [], {}
    for n in sorted((n for n in ncut if cut.get(n) is not None), key=lambda n: -cut[n]):
        text = reasoning[:cut[n]]
        x = ids(tok, text)
        assert x and tok.decode(x) == text, ("prefix does not round-trip", n)   # byte-identical to the 9B's prefix text
        for k, ch in enumerate(chains):
            if ch[:len(x)] == x:
                reads[n] = (k, len(x) - 1); break
        else:
            chains.append(x); reads[n] = (len(chains) - 1, len(x) - 1)
    return chains, reads
