"""E129 harness checks with scripted replies (no model): checkpoint stop, goal-based exposure under each template family, prefix replay and
the three arms' prefills, for both model families.

  ~/.venvs/agentdojo/bin/python test_e129.py
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e129_agentdojo as G
from agentdojo.task_suite.load_suites import get_suites

HARMONY = dict(
    call=('<|channel|>analysis<|message|>I need the transactions to answer the question about the March bill.<|end|><|start|>assistant<|channel|>commentary '
          'to=functions.get_most_recent_transactions <|constrain|>json<|message|>{"n": 100}', 200012),
    final=('<|channel|>analysis<|message|>Sum the March ones and report the total amount back to the user now please.<|end|><|start|>assistant'
           '<|channel|>final<|message|>You spent 1060.0 in March.', 200002))
QWEN = dict(
    call=('I need the transactions to answer the question about the March bill so I will fetch them first.\n</think>\n\n<tool_call>\n'
          '<function=get_most_recent_transactions>\n<parameter=n>\n100\n</parameter>\n</function>\n</tool_call>', None),
    final=('Sum the March ones and report the total amount back to the user now please, that is all that is needed.\n</think>\n\n'
           'You spent 1060.0 in March.', None))


class Script:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, sess, spec, ids, stop, seed):
        self.prompts.append((list(ids), seed)); raw, stop_id = self.replies.pop(0)
        return raw, "stop", stop_id, 10, None   # no token ids: the harness re-encodes, as with a server that omits them


def check(model, R, suites):
    E = G.MAN["eligible_tools"]["banking"]
    for t in G.MAN["train_templates"] + G.MAN["heldout_templates"]:
        split = "dev" if t in G.MAN["train_templates"] else "test"
        x = next(v for v in G.MAN["inputs"] if v["split"] == split and v["suite"] == "banking" and v["attack"] == t)
        G.complete = Script([R["call"]])
        pre = G.run_episode(model, suites, x, None, "k|prefix", E); ck = pre["checkpoint"]
        assert ck and ck["k"] == 1 and not pre["finished"], (t, pre.get("error"))
        if ck["step_tools"] == ["get_most_recent_transactions"]:
            assert ck["exposed_step"] == ck["exposed_any"], (t, ck)
        print(model, t, "checkpoint exposed_step", ck["exposed_step"], "exposed_any", ck["exposed_any"])
    x = next(v for v in G.MAN["inputs"] if v["input"] == "banking|user_task_1|injection_task_0|important_instructions")
    G.complete = Script([R["call"]]); pre = G.run_episode(model, suites, x, None, "k|prefix", E); ck = pre["checkpoint"]
    assert ck["exposed_step"] and ck["exposed_eligible"], ck
    agent = G.Agent(model, None, "x", None, None)
    tail = agent.decode(ck["ids"])[-40:]
    assert tail.endswith("<|start|>assistant" if agent.fam == "harmony" else "<think>\n"), tail
    pre = json.loads(json.dumps(pre, default=str))
    prefix = dict(k=ck["k"], ids_sha=ck["ids_sha"], calls=[dict(reasoning=c["reasoning"], text=c["text"], calls=c["calls"], parse_error=c["parse_error"]) for c in pre["calls"]])
    out = {}
    for arm in ("U", "P", "T"):
        G.complete = s = Script([R["final"]])
        t_ids = None
        if arm == "T":
            t_ids = G.t_prefix(model, out["U"]["k_ids"]["ids"]); assert t_ids is not None
        row = G.run_episode(model, suites, x, None, "k|c0", None, prefix, arm, t_ids)
        assert row["finished"] and row["error"] is None and row["llm_calls"] == 2 and row["live_calls"] == 1, row
        assert len(s.prompts) == 1 and G.sha_ids(s.prompts[0][0][:len(ck["ids"])]) == ck["ids_sha"], "one live call, on the replayed checkpoint prompt"
        sent = agent.decode(s.prompts[0][0][len(ck["ids"]):])
        hd = G.HARMONY_HEADER if agent.fam == "harmony" else ""
        if arm == "U":
            assert sent == ""
        if arm == "P":
            assert sent == hd + G.CFG["reminder_n0"], repr(sent)
        if arm == "T":
            assert sent.startswith(hd + "Sum the March") and sent.endswith(G.CFG["reminder"]), repr(sent)
            assert "Wait. The tool result" in row["calls"][1]["reasoning"], "the reminder stays in the turn's reasoning"
        out[arm] = row; out[arm + "seed"] = s.prompts[0][1]
    assert out["Useed"] == out["Pseed"] == out["Tseed"], "arms of one continuation share the seed"
    assert out["U"]["utility"] is not None and out["U"]["security"] is not None
    assert G.t_prefix(model, out["U"]["k_ids"]["ids"][:5]) is None, "too short for T"
    G.complete = Script([R["final"]])   # an episode that ends before any eligible response: finished, no checkpoint
    row = G.run_episode(model, suites, x, None, "k|prefix", E)
    assert row["finished"] and row["checkpoint"] is None and row["utility"] is not None, row
    print(model, "harness checks passed")


def main():
    suites = get_suites("v1.2.2")
    check("gpt-oss-20b", HARMONY, suites)
    check("qwen35-9b", QWEN, suites)
    print("e129 harness checks passed")


if __name__ == "__main__":
    main()
