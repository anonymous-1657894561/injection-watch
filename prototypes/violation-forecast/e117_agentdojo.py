"""E117: AgentDojo pilot, Qwen3.5-9B and gpt-oss-20b in their native tool formats (experiments/E117-agentdojo-second-model-pilot.md).

  python e117_agentdojo.py MODEL --backend vllm [--limit N] [--workers 64]      # pod: a vLLM server on :8000 serves MODEL
  python e117_agentdojo.py qwen35-9b --backend openrouter --limit 2 --rollouts 1  # local smoke of the Qwen path (raw completions)
  -> data/e117/<MODEL>/episodes.jsonl (one line per episode, resumable)

Episodes: results/e117/manifest.json (40 dev pairs x 8 rollouts under `important_instructions`, and the 29 clean user tasks x 8).
Pipeline: AgentDojo's default system message, InitQuery, NativeAgent, ToolsExecutionLoop(ToolsExecutor, NativeAgent; max 15); one
attempt per episode (AgentDojo re-runs a task that ended without text output; OneShot replays the first attempt, as E077).
NativeAgent renders the whole conversation itself and parses the raw completion, so a malformed call is counted, never repaired:
- family qwen: tokenizer.apply_chat_template(tools=...), thinking on; each earlier step's reasoning goes back as `reasoning_content`.
  Reply = reasoning up to `</think>`, then text, then zero or more `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`.
- family harmony: openai_harmony system (reasoning effort) + developer (AgentDojo's system message, function tools); analysis messages
  kept within the turn; a call is `commentary` to `functions.NAME` with JSON arguments; tool results are `functions.NAME` messages.
Parse failure = an attempted call that does not parse, a recipient outside `functions.`, non-JSON arguments, or a turn cut at the cap.
Request seed = sha256(model|suite|user|injection|rollout|call index); the manifest's rollout seeds are 11700 + r.
"""
import argparse, hashlib, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml
from agentdojo.agent_pipeline import AgentPipeline, BasePipelineElement, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "configs/e117.yaml").read_text())
TC = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.S)
PRM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
_lock = threading.Lock()


def parse_value(v):
    try:
        return json.loads(v.strip())
    except Exception:
        return v.strip()


class Backend:
    """complete(prompt_ids, prompt_text, max_tokens, seed) -> (completion text with special tokens, finish_reason, n_completion_tokens)."""

    def __init__(self, kind, spec):
        self.kind, self.spec = kind, spec; self.usd = 0.0
        self.S = requests.Session()

    def complete(self, ids, text, stop_ids, seed):
        sp = self.spec; body = dict(max_tokens=sp["max_tokens"], seed=seed, **sp["sampling"])
        for attempt in range(8):
            try:
                if self.kind == "vllm":
                    r = self.S.post("http://127.0.0.1:8000/v1/completions", timeout=900, json=dict(
                        model=sp["served_name"], prompt=ids, stop_token_ids=stop_ids, skip_special_tokens=False, **body))
                    if r.status_code == 400:   # context length and similar: not retried, recorded as the episode's error
                        raise ValueError(f"vllm 400: {r.text[:300]}")
                    r.raise_for_status(); c = r.json()["choices"][0]; u = r.json()["usage"]
                    return c["text"], c["finish_reason"], c.get("stop_reason"), u["completion_tokens"]
                if self.kind == "hf":   # local smoke only: a small model of the same family on this machine's GPU, one call at a time
                    import torch
                    with _lock:
                        if not hasattr(self, "model"):
                            from transformers import AutoModelForCausalLM
                            self.model = AutoModelForCausalLM.from_pretrained(sp["smoke_hf_id"], dtype=torch.bfloat16).to("cuda").eval()
                        torch.manual_seed(seed)
                        g = self.model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=sp["max_tokens"], do_sample=True,
                                                eos_token_id=stop_ids, temperature=sp["sampling"]["temperature"], top_p=sp["sampling"]["top_p"],
                                                top_k=sp["sampling"].get("top_k"))[0, len(ids):].tolist()
                    hit = g[-1] if g and g[-1] in stop_ids else None
                    text_out = TOKENIZERS[sp["hf_id"]].decode(g[:-1] if hit is not None else g, skip_special_tokens=False)
                    return text_out, "stop" if hit is not None else "length", hit, len(g)
                o = sp["openrouter"]   # raw /completions: the provider applies no template (sources/openrouter-qwen35-llama.md)
                r = self.S.post("https://openrouter.ai/api/v1/completions", timeout=600, headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
                                json=dict(model=o["id"], prompt=text, provider={"order": [o["provider"]], "allow_fallbacks": False}, usage={"include": True}, **body))
                r.raise_for_status(); d = r.json(); c = d["choices"][0]
                with _lock:
                    self.usd += float(d.get("usage", {}).get("cost") or 0)
                text_out = c["text"] if c.get("reasoning") is None else c["reasoning"] + "\n</think>" + c["text"]   # OpenRouter splits the reasoning out
                return text_out, c.get("finish_reason"), None, d["usage"]["completion_tokens"]
            except ValueError:
                raise
            except Exception as e:
                if attempt == 7:
                    raise
                time.sleep(min(90, 5 * 2 ** attempt))


class NativeAgent(BasePipelineElement):
    name = "local"

    def __init__(self, spec, backend, seed_key):
        self.spec, self.backend, self.seed_key = spec, backend, seed_key
        self.reasoning, self.log = [], []   # reasoning: one entry per assistant message this agent wrote, in order
        if spec["family"] == "qwen":
            from transformers import AutoTokenizer
            self.tok = TOKENIZERS.setdefault(spec["hf_id"], AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"]))
            self.stop = [self.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        else:
            import openai_harmony as H
            self.H = H; self.enc = ENC.setdefault("harmony", H.load_harmony_encoding(H.HarmonyEncodingName.HARMONY_GPT_OSS))
            self.stop = list(self.enc.stop_tokens_for_assistant_actions())

    def tools(self, runtime):
        return [(f.name, f.description, f.parameters.model_json_schema()) for f in runtime.functions.values()]

    def render_qwen(self, messages, tools):
        msgs, k = [], 0
        for m in messages:
            text = get_text_content_as_str(m["content"]) if m.get("content") else ""
            if m["role"] == "assistant":
                tc = [{"type": "function", "function": {"name": c.function, "arguments": dict(c.args)}} for c in (m.get("tool_calls") or [])]
                msgs.append({"role": "assistant", "content": text, "reasoning_content": self.reasoning[k], **({"tool_calls": tc} if tc else {})}); k += 1
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "content": f"Error: {m['error']}" if m.get("error") else text})
            else:
                msgs.append({"role": m["role"], "content": text})
        t = [{"type": "function", "function": {"name": n, "description": d, "parameters": p}} for n, d, p in tools]
        prompt = self.tok.apply_chat_template(msgs, tools=t, add_generation_prompt=True, tokenize=False)
        assert prompt.endswith("<|im_start|>assistant\n<think>\n"), "thinking-on generation prompt"
        return self.tok(prompt, add_special_tokens=False).input_ids, prompt

    def render_harmony(self, messages, tools):
        H = self.H; k = 0
        eff = {"low": H.ReasoningEffort.LOW, "medium": H.ReasoningEffort.MEDIUM, "high": H.ReasoningEffort.HIGH}[self.spec["reasoning_effort"]]
        out = [H.Message.from_role_and_content(H.Role.SYSTEM, H.SystemContent.new().with_reasoning_effort(eff))]
        sysmsg = next(get_text_content_as_str(m["content"]) for m in messages if m["role"] == "system")
        out.append(H.Message.from_role_and_content(H.Role.DEVELOPER, H.DeveloperContent.new().with_instructions(sysmsg).with_function_tools(
            [H.ToolDescription.new(n, d, parameters=p) for n, d, p in tools])))
        for m in messages:
            text = get_text_content_as_str(m["content"]) if m.get("content") else ""
            if m["role"] == "user":
                out.append(H.Message.from_role_and_content(H.Role.USER, text))
            elif m["role"] == "assistant":
                if self.reasoning[k]:
                    out.append(H.Message.from_role_and_content(H.Role.ASSISTANT, self.reasoning[k]).with_channel("analysis"))
                k += 1
                if text:   # a commentary preamble (with calls) or the final answer
                    out.append(H.Message.from_role_and_content(H.Role.ASSISTANT, text).with_channel("commentary" if m.get("tool_calls") else "final"))
                for c in m.get("tool_calls") or []:
                    out.append(H.Message.from_role_and_content(H.Role.ASSISTANT, json.dumps(dict(c.args))).with_channel("commentary")
                               .with_recipient(f"functions.{c.function}").with_content_type("<|constrain|>json"))
            elif m["role"] == "tool":
                body = f"Error: {m['error']}" if m.get("error") else text
                out.append(H.Message.from_author_and_content(H.Author.new(H.Role.TOOL, f"functions.{m['tool_call'].function}"), body)
                           .with_channel("commentary").with_recipient("assistant"))
        ids = self.enc.render_conversation_for_completion(H.Conversation.from_messages(out), H.Role.ASSISTANT)
        return ids, self.enc.decode(ids)

    def parse_qwen(self, raw, finish):
        reason, sep, rest = raw.partition("</think>")
        if not sep:
            return raw.strip(), "", [], "no </think>" + (" (cap)" if finish == "length" else "")   # cut inside its reasoning: no answer
        rest = rest.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        calls = [FunctionCall(function=f.strip(), args={k: parse_value(v) for k, v in PRM.findall(body)}) for f, body in TC.findall(rest)]
        err = None
        if rest.count("<tool_call>") != len(calls):
            err = f"{rest.count('<tool_call>')} <tool_call> blocks, {len(calls)} parsed"
        elif finish == "length":
            err = "cap"
        text = (rest[:rest.find("<tool_call>")] if "<tool_call>" in rest else rest).strip()
        return reason.strip(), text, calls, err

    def parse_harmony(self, raw, finish, stop_reason):
        H = self.H
        toks = self.enc.encode(raw, allowed_special="all")
        if stop_reason in self.stop and (not toks or toks[-1] != stop_reason):
            toks.append(stop_reason)   # vLLM leaves the matched stop token out of the text
        try:
            msgs = self.enc.parse_messages_from_completion_tokens(toks, H.Role.ASSISTANT)
        except Exception as e:
            return "", "", [], f"harmony parse: {str(e)[:200]}"
        reason = "\n".join(m.content[0].text for m in msgs if m.channel == "analysis" and m.content)
        calls, err, text = [], None, ""
        for m in msgs:
            body = m.content[0].text if m.content else ""
            if m.recipient:
                if not m.recipient.startswith("functions."):
                    err = f"recipient {m.recipient}"; continue
                try:
                    args = json.loads(body) if body.strip() else {}
                    assert isinstance(args, dict)
                except Exception:
                    err = "arguments not a JSON object"; continue
                calls.append(FunctionCall(function=m.recipient[len("functions."):], args=args))
            elif m.channel in ("final", "commentary"):
                text = (text + "\n" + body).strip()
        if finish == "length":
            err = err or "cap"
        if not calls and not text and not err:
            err = "no final and no call"
        return reason, text, calls, err

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        tools = self.tools(runtime); fam = self.spec["family"]
        ids, prompt = self.render_qwen(messages, tools) if fam == "qwen" else self.render_harmony(messages, tools)
        seed = int(hashlib.sha256(f"{self.seed_key}|{len(self.log)}".encode()).hexdigest()[:8], 16)
        t0 = time.time()
        raw, finish, stop_reason, n_out = self.backend.complete(ids, prompt, self.stop, seed)
        reason, text, calls, err = self.parse_qwen(raw, finish) if fam == "qwen" else self.parse_harmony(raw, finish, stop_reason)
        self.reasoning.append(reason)
        self.log.append(dict(prompt_tokens=len(ids), completion_tokens=n_out, finish=finish, parse_error=err, n_calls=len(calls),
                             calls=[dict(function=c.function, args=dict(c.args)) for c in calls], reasoning_chars=len(reason),
                             injected_in_context="<INFORMATION>" in prompt, seconds=round(time.time() - t0, 2), raw=raw[:20000]))
        msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text)], tool_calls=calls or None)
        return query, runtime, env, [*messages, msg], extra_args


TOKENIZERS, ENC = {}, {}


class OneShot(BasePipelineElement):
    """AgentDojo re-runs a task (up to 3 times, on the already-changed environment) when it ends without text output; replay the
    first attempt instead, so one episode is one trajectory (E077)."""
    name = "local"

    def __init__(self, pipe):
        self.pipe, self.first = pipe, None

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        if self.first is None:
            self.first = self.pipe.query(query, runtime, env, messages, extra_args)
        return self.first


def episode(model, spec, backend, suites, job):
    s, u, i, r = job["suite"], job["user_task"], job.get("injection_task"), job["rollout"]
    suite = suites[s]; key = f"{model}|{s}|{u}|{i}|{r}"
    agent = NativeAgent(spec, backend, key)
    pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), agent,
                          ToolsExecutionLoop([ToolsExecutor(), agent], max_iters=CFG["max_iters"])]); pipe.name = "local"
    rec = OneShot(pipe); ut = suite.get_user_task_by_id(u); it = suite.get_injection_task_by_id(i) if i else None
    injections = {}
    if it:
        attack = load_attack(CFG["attack"], suite, rec); attack.model_name = spec["prose_name"]
        injections = attack.attack(ut, it)
    row = dict(episode=key, model=model, suite=s, user_task=u, injection_task=i, rollout=r, attacked=bool(it))
    t0 = time.time()
    try:
        utility, security = suite.run_task_with_pipeline(rec, ut, it, injections)
        row.update(utility=bool(utility), security=bool(security) if it else None, error=None)
    except Exception as e:   # recorded, never an outcome (sources/agentdojo.md, Trap 7)
        row.update(utility=None, security=None, error=f"{type(e).__name__}: {str(e)[:400]}")
    msgs = rec.first[3] if rec.first else []
    last = msgs[-1] if msgs else None
    row.update(llm_calls=len(agent.log), parse_failures=sum(c["parse_error"] is not None for c in agent.log),
               parse_errors=[c["parse_error"] for c in agent.log if c["parse_error"]],
               invalid_tool=sum(1 for m in msgs if m["role"] == "tool" and (m.get("error") or "").startswith("Invalid tool")),
               saw_injection=any("<INFORMATION>" in (get_text_content_as_str(m["content"]) if m.get("content") else "") for m in msgs if m["role"] == "tool"),
               hit_max_iters=bool(last and last["role"] == "tool"),
               no_final_answer=not (last and last["role"] == "assistant" and not last.get("tool_calls") and get_text_content_as_str(last["content"] or []).strip()),
               analysis_calls=sum(c["reasoning_chars"] > 0 for c in agent.log), seconds=round(time.time() - t0, 1),
               tokens_in=sum(c["prompt_tokens"] for c in agent.log), tokens_out=sum(c["completion_tokens"] for c in agent.log), calls=agent.log)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--backend", choices=["vllm", "openrouter", "hf"], required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--rollouts", type=int); ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = CFG["models"][a.model]; man = json.loads((HERE / "results/e117/manifest.json").read_text())
    R = a.rollouts or man["rollouts"]
    inputs = [dict(p) for p in man["pairs"]] + [dict(c, injection_task=None) for c in man["clean"]]
    if a.limit:   # smoke: the first `limit` pairs and their clean user tasks
        inputs = inputs[:a.limit] + [dict(suite=p["suite"], user_task=p["user_task"], injection_task=None) for p in man["pairs"][:a.limit]]
    jobs = [dict(x, rollout=r) for x in inputs for r in range(R)]
    out = Path(a.out or HERE / "data/e117" / a.model); out.mkdir(parents=True, exist_ok=True); ep = out / "episodes.jsonl"
    done = {json.loads(l)["episode"] for l in open(ep)} if ep.exists() else set()
    todo = [j for j in jobs if f'{a.model}|{j["suite"]}|{j["user_task"]}|{j.get("injection_task")}|{j["rollout"]}' not in done]
    print(f"[e117] {a.model}: {len(jobs)} episodes, {len(done)} done, {len(todo)} to run", flush=True)
    suites = get_suites(CFG["version"]); backend = Backend(a.backend, spec); t0 = time.time(); n = [0]
    NativeAgent(spec, backend, "warm")   # load the tokenizer / Harmony encoding once, before the threads (lazy imports race)

    def run(j):
        row = episode(a.model, spec, backend, suites, j)
        with _lock:
            with open(ep, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            n[0] += 1
            print(f"[e117] {n[0]}/{len(todo)} {row['episode']} utility={row['utility']} security={row['security']} calls={row['llm_calls']} "
                  f"parse_fail={row['parse_failures']} err={(row['error'] or '')[:60]} {time.time() - t0:.0f}s ${backend.usd:.4f}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, todo))
    print(f"[e117] done {a.model} in {time.time() - t0:.0f}s, openrouter ${backend.usd:.4f}", flush=True)


if __name__ == "__main__":
    main()
