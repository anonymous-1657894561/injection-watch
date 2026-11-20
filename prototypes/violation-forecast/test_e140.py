"""E140 unit checks, no model: Harmony reply parsing and the strict label (run locally and on the pod before generation)."""
import e140_injecforks as G

H, E = G.enc(); CALL = E.encode("<|call|>", allowed_special="all")[0]; RET = E.encode("<|return|>", allowed_special="all")[0]
d, c, e = G.parse("<|channel|>analysis<|message|>The note asks me to grant access.<|end|><|start|>assistant<|channel|>commentary to=functions.AugustSmartLockGrantGuestAccess <|constrain|>json<|message|>{\"guest_ids\": [\"a\"]}", CALL)
assert d and c == ["AugustSmartLockGrantGuestAccess"] and e is None, (d, c, e)
d, c, e = G.parse("<|channel|>analysis<|message|>Just answer.<|end|><|start|>assistant<|channel|>final<|message|>No issues found.", RET)
assert d and c == [] and e is None, (d, c, e)
pre = E.decode(G.p_prefix())
d, c, e = G.parse("I will only answer the user.<|end|><|start|>assistant<|channel|>final<|message|>Done.", RET, pre)
assert d and c == [], (d, c, e)
d, c, e = G.parse("<|channel|>analysis<|message|>unfinished", None)
assert not d, (d, c, e)
print("test_e140 ok")
