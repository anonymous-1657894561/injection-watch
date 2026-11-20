"""E125 harness checks with scripted replies (no model): checkpoint stop, prefix replay, the three arms' prefills.

  ~/.venvs/agentdojo/bin/python test_e125.py
"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e125_agentdojo as G
from agentdojo.task_suite.load_suites import get_suites

CALL = ('<|channel|>analysis<|message|>I need the transactions to answer the question about the March bill.<|end|><|start|>assistant<|channel|>commentary '
        'to=functions.get_most_recent_transactions <|constrain|>json<|message|>{"n": 100}', 200012)
FINAL = ('<|channel|>analysis<|message|>Sum the March ones and report the total amount back to the user now please.<|end|><|start|>assistant'
         '<|channel|>final<|message|>You spent 1060.0 in March.', 200002)


class Script:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, sess, ids, stop, seed):
        self.prompts.append((list(ids), seed)); raw, stop_id = self.replies.pop(0)
        return raw, "stop", stop_id, 10, None   # no token ids: the harness re-encodes, as with a server that omits them


def main():
    suites = get_suites("v1.2.2"); E = G.MAN["eligible_tools"]["banking"]
    x = next(v for v in G.MAN["inputs"] if v["input"] == "banking|user_task_1|injection_task_0" and v["split"] == "dev")
    G.A.NativeAgent(G.SPEC, None, "warm"); enc = G.A.ENC["harmony"]
    G.complete = s = Script([CALL])
    pre = G.run_episode(suites, x, None, "k|prefix", E)
    ck = pre["checkpoint"]
    assert ck and ck["k"] == 1 and len(pre["calls"]) == 1 and not pre["finished"], pre
    assert ck["step_tools"] == ["get_most_recent_transactions"] and ck["exposed_step"] and ck["exposed_eligible"], ck
    assert enc.decode(ck["ids"]).endswith("<|start|>assistant") and "<tool_result name=\"get_most_recent_transactions\">" in ck["transcript"]
    pre = json.loads(json.dumps(pre, default=str))
    prefix = dict(k=ck["k"], ids_sha=ck["ids_sha"], calls=[dict(reasoning=c["reasoning"], text=c["text"], calls=c["calls"], parse_error=c["parse_error"]) for c in pre["calls"]])
    out = {}
    for arm in ("U", "P", "T"):
        G.complete = s = Script([FINAL])
        t_ids = None
        if arm == "T":
            ki = out["U"]["k_ids"]["ids"]; hd = enc.encode(G.HEADER, allowed_special="all")
            assert ki[:len(hd)] == hd and len(ki) >= len(hd) + 16; t_ids = ki[:len(hd) + 16]
        row = G.run_episode(suites, x, None, "k|c0", None, prefix, arm, t_ids)
        assert row["finished"] and row["error"] is None and row["llm_calls"] == 2 and row["live_calls"] == 1, row
        assert len(s.prompts) == 1 and G.sha_ids(s.prompts[0][0][:len(ck["ids"])]) == ck["ids_sha"], "one live call, on the replayed checkpoint prompt"
        sent = enc.decode(s.prompts[0][0][len(ck["ids"]):])
        if arm == "U":
            assert sent == ""
        if arm == "P":
            assert sent == G.HEADER + G.CFG["reminder_n0"], sent
        if arm == "T":
            assert sent.startswith(G.HEADER + "Sum the March") and sent.endswith(G.CFG["reminder"]), sent
            assert "Wait. The tool result" in row["calls"][1]["reasoning"], "the reminder stays in the turn's analysis"
        out[arm] = row; out[arm + "seed"] = s.prompts[0][1]
    assert out["Useed"] == out["Pseed"] == out["Tseed"], "arms of one continuation share the seed"
    assert out["U"]["utility"] is not None and out["U"]["security"] is not None
    G.complete = Script([FINAL])   # an episode that ends before any eligible response: finished, no checkpoint
    row = G.run_episode(suites, x, None, "k|prefix", E)
    assert row["finished"] and row["checkpoint"] is None and row["utility"] is not None, row
    G.complete = Script([FINAL])
    row = G.run_episode(suites, x, None, "k|c0", None)   # its continuations: fresh untreated episodes
    assert row["finished"] and row["live_calls"] == 1, row
    print("e125 harness checks passed")


if __name__ == "__main__":
    main()
