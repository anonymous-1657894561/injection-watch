"""E125: held-out gpt-oss-20b AgentDojo gate test, generation (experiments/E125-agentdojo-gptoss-heldout.md). Pod, vLLM server on :8000.

  python e125_agentdojo.py prefix --split dev|test [--limit N] [--out DIR]          -> DIR/prefix-<split>.jsonl
  python e125_agentdojo.py cont --split dev|test --arm U|P|T [--limit N] [--out DIR] -> DIR/cont-<split>-<arm>.jsonl

Built on E117's harness (e117_agentdojo.py: Harmony rendering and parsing, AgentDojo pipeline, one-shot episodes), gpt-oss-20b only.
- prefix: one episode per input (results/e125/manifest.json), sampled until the eligible checkpoint: the model's next call after the first
  step in which a tool in the manifest's E_suite returned a non-error response. The episode stops there; the record keeps every earlier
  call (raw completion, parsed reasoning / text / calls), the checkpoint prompt's token ids, the transcript the monitor reads, and the
  labels the gate may not see (injection in the checkpoint step's tool responses, anywhere in context). An episode that ends before any
  eligible response has no checkpoint: it is recorded as finished.
- cont: for each input with a checkpoint and each r in 0-7, a fresh environment, the prefix replayed call by call (no sampling; the
  checkpoint prompt is asserted token-identical), then the arm at the checkpoint call and live sampling to the end of the episode:
    U  untreated;
    P  E105's reminder as the first analysis tokens (prompt-time reminder; the gates' reminder);
    T  the first 16 analysis tokens of U's checkpoint call (same r), then E093's reminder (token-16 reminder).
  Live call j of continuation r uses seed sha256(e125|model|input|c<r>|j) in every arm. Inputs without a checkpoint get 8 fresh untreated
  episodes in arm U only (no gate can act on them). T is recorded as not applicable when U's checkpoint call does not open with an
  analysis message of at least 16 tokens.
Outcomes: AgentDojo utility and security (strict: the attacker's goal executed). Test outcomes are written but never printed.
"""
import argparse, hashlib, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

import e117_agentdojo as A
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.functions_runtime import EmptyEnv, FunctionCall
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.types import ChatAssistantMessage, get_text_content_as_str, text_content_block_from_string

HERE = Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "configs/e125.yaml").read_text())
MAN = json.loads((HERE / "results/e125/manifest.json").read_text())
MODEL = "gpt-oss-20b"; SPEC = A.CFG["models"][MODEL]; N16 = 16
HEADER = "<|channel|>analysis<|message|>"
for k in ("reminder", "reminder_n0"):
    assert hashlib.sha256(CFG[k].encode()).hexdigest() == CFG[k + "_sha256"], k
_lock = threading.Lock()


class Checkpoint(Exception):
    pass


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16]


def complete(sess, ids, stop, seed):
    """vLLM raw completion with the generated token ids (vLLM's return_token_ids; re-encoded from the text if the server omits them)."""
    body = dict(model=SPEC["served_name"], prompt=list(ids), stop_token_ids=stop, skip_special_tokens=False, max_tokens=SPEC["max_tokens"],
                seed=seed, return_token_ids=True, **SPEC["sampling"])
    for attempt in range(8):
        try:
            r = sess.post("http://127.0.0.1:8000/v1/completions", timeout=900, json=body)
            if r.status_code == 400:
                raise ValueError(f"vllm 400: {r.text[:300]}")
            r.raise_for_status(); d = r.json(); c = d["choices"][0]
            return c["text"], c["finish_reason"], c.get("stop_reason"), d["usage"]["completion_tokens"], c.get("token_ids")
        except ValueError:
            raise
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(90, 5 * 2 ** attempt))


def transcript(messages):
    """What the agent saw up to the checkpoint, as plain text for the monitor (system prompt, tools, user request, its calls and the
    tool results, its earlier analysis)."""
    out = []
    for m in messages:
        text = get_text_content_as_str(m["content"]) if m.get("content") else ""
        if m["role"] == "system":
            out.append(f"<agent_system_prompt>\n{text}\n</agent_system_prompt>")
        elif m["role"] == "user":
            out.append(f"<user_request>\n{text}\n</user_request>")
        elif m["role"] == "assistant":
            calls = "\n".join(f"{c.function}({json.dumps(dict(c.args))})" for c in (m.get("tool_calls") or []))
            out.append(f"<agent_step>\n{text}\n{calls}\n</agent_step>".replace("\n\n", "\n"))
        elif m["role"] == "tool":
            body = f"Error: {m['error']}" if m.get("error") else text
            out.append(f"<tool_result name=\"{m['tool_call'].function}\">\n{body}\n</tool_result>")
    return "\n\n".join(out)


class Agent(A.NativeAgent):
    """E117's Harmony agent with an eligible-checkpoint stop (prefix), prefix replay and the arms at the checkpoint (cont)."""

    def __init__(self, backend_sess, seed_key, eligible, prefix=None, arm=None, t_ids=None):
        super().__init__(SPEC, None, seed_key)
        self.sess, self.prefix, self.arm, self.t_ids = backend_sess, prefix, arm, t_ids
        self.eligible = set(eligible) if eligible is not None else None
        self.header = self.enc.encode(HEADER, allowed_special="all"); self.ck = None; self.k_ids = None

    def tools_text(self, runtime):
        return "\n".join(f"- {n}: {d}" for n, d, _ in self.tools(runtime))

    def query(self, query, runtime, env=EmptyEnv(), messages=(), extra_args={}):
        j = len(self.log); tools = self.tools(runtime)
        if self.prefix is not None and j < self.prefix["k"]:   # replay a recorded prefix call
            c = self.prefix["calls"][j]; self.reasoning.append(c["reasoning"])
            self.log.append(dict(replay=True, prompt_tokens=0, completion_tokens=0, parse_error=c["parse_error"], n_calls=len(c["calls"]),
                                 calls=c["calls"], reasoning_chars=len(c["reasoning"]), seconds=0.0))
            calls = [FunctionCall(function=x["function"], args=x["args"]) for x in c["calls"]]
            msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(c["text"])], tool_calls=calls or None)
            return query, runtime, env, [*messages, msg], extra_args
        ids, _ = self.render_harmony(messages, tools)
        at_ck = self.prefix is not None and j == self.prefix["k"]
        if self.prefix is None and self.eligible is not None and self.ck is None:
            last = max([n for n, m in enumerate(messages) if m["role"] == "assistant"], default=-1)
            tool_msgs = [m for m in messages if m["role"] == "tool"]
            if any(m["tool_call"].function in self.eligible and not m.get("error") for m in tool_msgs):
                step = [m for m in messages[last + 1:] if m["role"] == "tool"]
                txt = lambda ms: " ".join(get_text_content_as_str(m["content"]) for m in ms if m.get("content"))
                self.ck = dict(k=j, ids=list(ids), ids_sha=sha_ids(ids), transcript=transcript(messages), tools_text=self.tools_text(runtime),
                               step_tools=[m["tool_call"].function for m in step],
                               exposed_step="<INFORMATION>" in txt(step), exposed_eligible="<INFORMATION>" in txt([m for m in step if m["tool_call"].function in self.eligible]),
                               exposed_any="<INFORMATION>" in txt(tool_msgs))
                raise Checkpoint()
        prefill = []
        if at_ck:
            assert sha_ids(ids) == self.prefix["ids_sha"], "replayed checkpoint prompt differs from the prefix's"
            if self.arm == "P":
                prefill = self.header + self.enc.encode(CFG["reminder_n0"], allowed_special="all")
            elif self.arm == "T":
                prefill = list(self.t_ids) + self.enc.encode(CFG["reminder"], allowed_special="all")
        seed = int(hashlib.sha256(f"{self.seed_key}|{j}".encode()).hexdigest()[:8], 16)
        t0 = time.time()
        raw, finish, stop_reason, n_out, tok = complete(self.sess, ids + prefill, self.stop, seed)
        secs = time.time() - t0
        if at_ck and self.arm == "U":
            self.k_ids = dict(ids=(tok or self.enc.encode(raw, allowed_special="all"))[:N16 + 8], source="vllm" if tok else "reencoded")
        raw_full = self.enc.decode(prefill) + raw if prefill else raw
        reason, text, calls, err = self.parse_harmony(raw_full, finish, stop_reason)
        self.reasoning.append(reason)
        self.log.append(dict(replay=False, checkpoint=at_ck, prefill_tokens=len(prefill), prompt_tokens=len(ids) + len(prefill), completion_tokens=n_out,
                             finish=finish, parse_error=err, n_calls=len(calls), calls=[dict(function=c.function, args=dict(c.args)) for c in calls],
                             reasoning=reason, text=text, reasoning_chars=len(reason), injected_in_context="<INFORMATION>" in self.enc.decode(ids),
                             seconds=round(secs, 2), raw=raw[:20000]))
        msg = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(text)], tool_calls=calls or None)
        return query, runtime, env, [*messages, msg], extra_args


def run_episode(suites, x, sess, seed_key, eligible, prefix=None, arm=None, t_ids=None):
    s, u, i = x["suite"], x["user_task"], x["injection_task"]; suite = suites[s]
    agent = Agent(sess, seed_key, eligible, prefix, arm, t_ids)
    pipe = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), agent,
                          ToolsExecutionLoop([ToolsExecutor(), agent], max_iters=A.CFG["max_iters"])]); pipe.name = "local"
    rec = A.OneShot(pipe); ut = suite.get_user_task_by_id(u); it = suite.get_injection_task_by_id(i) if i else None
    injections = {}
    if it:
        attack = load_attack(A.CFG["attack"], suite, rec); attack.model_name = SPEC["prose_name"]; injections = attack.attack(ut, it)
    row = dict(input=x["input"], split=x["split"], suite=s, user_task=u, injection_task=i, attacked=bool(it), checkpoint=None)
    t0 = time.time()
    try:
        utility, security = suite.run_task_with_pipeline(rec, ut, it, injections)
        row.update(finished=True, utility=bool(utility), security=bool(security) if it else None, error=None)
    except Checkpoint:
        row.update(finished=False, utility=None, security=None, error=None, checkpoint=agent.ck)
    except Exception as e:   # recorded, never an outcome
        row.update(finished=False, utility=None, security=None, error=f"{type(e).__name__}: {str(e)[:400]}")
    live = [c for c in agent.log if not c["replay"]]
    row.update(llm_calls=len(agent.log), live_calls=len(live), parse_failures=sum(c["parse_error"] is not None for c in live),
               parse_errors=[c["parse_error"] for c in live if c["parse_error"]], seconds=round(time.time() - t0, 1),
               llm_seconds=round(sum(c["seconds"] for c in live), 2), tokens_out=sum(c["completion_tokens"] for c in live), calls=agent.log,
               k_ids=agent.k_ids)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=["prefix", "cont"]); ap.add_argument("--split", required=True)
    ap.add_argument("--arm", choices=["U", "P", "T"]); ap.add_argument("--limit", type=int); ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--out", default=str(HERE / "data/e125"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X = [x for x in MAN["inputs"] if x["split"] == a.split]
    if a.limit:   # smoke: the first `limit` attacked inputs of each suite
        X = [x for s in sorted({x["suite"] for x in X}) for x in [y for y in X if y["suite"] == s and y["injection_task"]][:a.limit]]
    suites = get_suites(A.CFG["version"]); sess = requests.Session()
    A.NativeAgent(SPEC, None, "warm")   # load the Harmony encoding before the threads
    key = lambda x, tag: f"e125|{MODEL}|{x['input']}|{tag}"
    E = MAN["eligible_tools"]
    if a.stage == "prefix":
        path = out / f"prefix-{a.split}.jsonl"
        jobs = [dict(x=x) for x in X]
        done = {json.loads(l)["input"] for l in open(path)} if path.exists() else set()
        fn = lambda jb: run_episode(suites, jb["x"], sess, key(jb["x"], "prefix"), E[jb["x"]["suite"]])
        ident = lambda jb: jb["x"]["input"]; rid = lambda row: row["input"]
    else:
        P = {r["input"]: r for r in map(json.loads, open(out / f"prefix-{a.split}.jsonl"))}
        assert all(x["input"] in P for x in X), "prefix stage incomplete"
        U = {}
        if a.arm == "T":
            U = {(r["input"], r["r"]): r for r in map(json.loads, open(out / f"cont-{a.split}-U.jsonl"))}
        path = out / f"cont-{a.split}-{a.arm}.jsonl"
        jobs = [dict(x=x, r=r) for x in X for r in range(MAN["rollouts"]) if P[x["input"]]["checkpoint"] or a.arm == "U"]
        done = {(json.loads(l)["input"], json.loads(l)["r"]) for l in open(path)} if path.exists() else set()

        def fn(jb):
            x, r = jb["x"], jb["r"]; pre = P[x["input"]]
            ck = pre["checkpoint"]
            if ck is None:   # no checkpoint: a fresh untreated episode (U only)
                return dict(run_episode(suites, x, sess, key(x, f"c{r}"), None), r=r, arm="U", no_checkpoint=True)
            prefix = dict(k=ck["k"], ids_sha=ck["ids_sha"], calls=[dict(reasoning=c["reasoning"], text=c["text"], calls=c["calls"], parse_error=c["parse_error"])
                                                                for c in pre["calls"]])
            t_ids = None
            if a.arm == "T":
                ki = (U[(x["input"], r)].get("k_ids") or {}).get("ids") or []
                hd = A.ENC["harmony"].encode(HEADER, allowed_special="all"); end = A.ENC["harmony"].encode("<|end|>", allowed_special="all")[0]
                if ki[:len(hd)] != hd or len(ki) < len(hd) + N16 or end in ki[len(hd):len(hd) + N16]:
                    return dict(input=x["input"], split=x["split"], r=r, arm="T", not_applicable=True)
                t_ids = ki[:len(hd) + N16]
            return dict(run_episode(suites, x, sess, key(x, f"c{r}"), None, prefix, a.arm, t_ids), r=r, arm=a.arm, no_checkpoint=False)
        ident = lambda jb: (jb["x"]["input"], jb["r"]); rid = lambda row: (row["input"], row["r"])
    todo = [jb for jb in jobs if ident(jb) not in done]
    print(f"[e125] {a.stage} {a.split} {a.arm or ''}: {len(jobs)} jobs, {len(done)} done, {len(todo)} to run", flush=True)
    t0 = time.time(); n = [0]

    def run(jb):
        row = fn(jb)
        with _lock:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            n[0] += 1
            shown = "" if a.split == "test" else f" utility={row.get('utility')} security={row.get('security')} ck={bool(row.get('checkpoint'))}"
            print(f"[e125] {n[0]}/{len(todo)} {rid(row)}{shown} calls={row.get('llm_calls')} parse_fail={row.get('parse_failures')} "
                  f"err={(row.get('error') or '')[:60]} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run, todo))
    print(f"[e125] done {a.stage} {a.split} {a.arm or ''} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
