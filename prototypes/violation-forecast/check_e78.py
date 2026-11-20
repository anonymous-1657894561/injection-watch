"""Structural checker for E78's hand-written InjecAgent user cases (experiments/E78-injecagent-new-user-tools.md). No LLM; exits 1.

Usage: python3 -I check_e78.py [configs/e78_user_cases.json]

Per user case: the tool exists in InjecAgent's tools.json, is not one of its 17 user tools or any attacker tool, and is used once;
the parameters parse, name only the tool's parameters, include every required one and have the declared JSON types; the response
template parses, its top-level keys are declared returns, and it holds exactly one injection slot, as a whole string value; the
Thought starts "Thought: " and names the tool. Crossed with the 62 attacker cases, every case places the injection exactly once.
Surface features (template length, slot depth, instruction length) are printed beside InjecAgent's own 17 and flagged when
outside their range.
"""
import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "data/e77/injecagent/raw/repo/data"   # fetched by e77_injecagent.py
SLOT = "<Attacker Instruction>"
TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "array": (list,), "object": (dict,)}

cases = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "configs/e78_user_cases.json"))
tools = {k["toolkit"] + t["name"]: t for k in json.load(open(RAW / "tools.json")) for t in k["tools"]}
orig = [json.loads(l) for l in open(RAW / "user_cases.jsonl")]
attackers = [json.loads(l) for f in ["attacker_cases_dh.jsonl", "attacker_cases_ds.jsonl"] for l in open(RAW / f)]
attacker_tools = {t for a in attackers for t in a["Attacker Tools"]}
errors = []


def slot_paths(x, path=()):
    if isinstance(x, str):
        return [path] if SLOT in x else []
    items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else []
    return [p for k, v in items for p in slot_paths(v, path + (k,))]


def value_at(x, path):
    for k in path:
        x = x[k]
    return x


def surface(c):
    t = ast.literal_eval(c["Tool Response Template"])
    return dict(template_chars=len(c["Tool Response Template"]), slot_depth=len(slot_paths(t)[0]), instruction_chars=len(c["User Instruction"]))


seen = set()
for c in cases:
    name = c["User Tool"]
    err = lambda m: errors.append(f"{name}: {m}")
    if name not in tools:
        err("not in tools.json"); continue
    if name in {o["User Tool"] for o in orig}:
        err("already one of InjecAgent's user tools")
    if name in attacker_tools:
        err("is an attacker tool")
    if name in seen:
        err("used twice")
    seen.add(name)
    spec = {p["name"]: p for p in tools[name]["parameters"]}
    args = ast.literal_eval(c["Tool Parameters"])
    for k, v in args.items():
        if k not in spec:
            err(f"parameter {k!r} is not the tool's")
        elif not isinstance(v, TYPES[spec[k]["type"]]) or (spec[k]["type"] in ("integer", "number") and isinstance(v, bool)):
            err(f"parameter {k!r} is {type(v).__name__}, declared {spec[k]['type']}")
    for k, p in spec.items():
        if p.get("required") and k not in args:
            err(f"required parameter {k!r} missing")
    tpl = ast.literal_eval(c["Tool Response Template"])
    returns = {r["name"] for r in tools[name].get("returns", [])}
    if isinstance(tpl, dict) and not set(tpl) <= returns:
        err(f"response keys {sorted(set(tpl) - returns)} are not declared returns")
    paths = slot_paths(tpl)
    if c["Tool Response Template"].count(SLOT) != 1 or len(paths) != 1:
        err("the injection slot is not there exactly once")
    elif value_at(tpl, paths[0]) != SLOT:
        err("the slot is not a whole string value")
    if not c["Thought"].startswith("Thought: ") or f'"{name}"' not in c["Thought"]:
        err("Thought does not start 'Thought: ' or does not name the tool in quotes")
    for a in attackers:
        if name in a["Attacker Tools"]:
            continue
        filled = c["Tool Response Template"].replace(SLOT, a["Attacker Instruction"])
        if filled.count(a["Attacker Instruction"]) != 1:
            err(f"injection placed {filled.count(a['Attacker Instruction'])} times for {a['Attacker Instruction'][:40]!r}")

ref = [surface(o) for o in orig]
print(f"{len(cases)} user cases, {len(cases) * len(attackers)} cases with the attacker cases")
for k in ref[0]:
    lo, hi = min(r[k] for r in ref), max(r[k] for r in ref)
    new = [surface(c)[k] for c in cases if c["User Tool"] in tools]
    out = [c["User Tool"] for c in cases if c["User Tool"] in tools and not lo <= surface(c)[k] <= hi]
    print(f"  {k}: InjecAgent {lo}-{hi} (mean {sum(r[k] for r in ref) / len(ref):.0f}); new mean {sum(new) / len(new):.0f}"
          + (f"; outside the range: {', '.join(out)}" if out else ""))
if errors:
    print("FAIL\n  " + "\n  ".join(errors))
    sys.exit(1)
print("PASS")
