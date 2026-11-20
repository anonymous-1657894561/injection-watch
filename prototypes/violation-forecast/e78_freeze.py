"""Freeze InjecForks v1 from the E78 v2 rollout files (experiments/E78-injecagent-new-user-tools.md, runbook steps 4-5).
LLM-free and API-free. Checks every rollout file, prints the Results tables, and with --stage writes the Hub layout:

  data/e78/injecforks_v1/<config>/<split>.jsonl          the rollout file, byte for byte
  data/e78/injecforks_v1/prompts/<config>/<split>.jsonl  one exact request body per case (e77_variants.py --dry --export)
  data/e78/injecforks_v1/cuts/<config>/<split>.jsonl     per row, the character offset in `reasoning` after the first N tokens
  data/e78/injecforks_v1/eval/<config>/<split>.jsonl     the shared evaluation subset (dev, test): rows every method scores
  data/e78/injecforks_v1/SHA256SUMS

Configs: smoke, realistic, smoke-benign, realistic-benign. Usage: python e78_freeze.py [--stage]
"""
import hashlib, json, random, shutil, subprocess, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
D, R = HERE / "data/e78/variants", HERE / "results/e78/variants"
OUT = HERE / "data/e78/injecforks_v1"
NS = [0, 8, 16, 32, 64, 128]
K = {"train": 8, "dev": 8, "test": 16}
CASES = {"train": 867, "dev": 186, "test": 1364}
BENIGN_CASES = {"train": 112, "dev": 24, "test": 176}
EST = {("smoke", "train"): 1.2, ("smoke", "dev"): 0.3, ("smoke", "test"): 3.1,      # runbook table, USD for new calls
       ("realistic", "train"): 4.3, ("realistic", "dev"): 0.9, ("realistic", "test"): 11.8}
ARM = {"smoke": ("tool_knowledge", "e78v2_tool_knowledge_qwen_thought", []),
       "realistic": ("important_no_names", "e78v2_important_no_names_qwen_thought_real", ["--realistic", "--cache=cache_real.jsonl"])}
SURFACE = {  # hand-assigned (E78 pre-build step 3; e78_tool_baseline.py and e77_variants.py's CARRIER)
    "search": ["GoogleSearchWebSearch", "GoogleSearchNewsSearch", "GoogleSearchVideoSearch", "GoogleMapGetLocationReviews",
               "InventoryManagementSystemSearchSuppliers", "AmazonSearchProducts", "TeladocSearchDoctors", "ExpediaSearchAccommodations",
               "IFTTTSearchApplets", "VenmoSearchTransactions", "EvernoteManagerSearchNotes", "GitHubSearchRepositories",
               "TwitterManagerSearchTweets", "TeladocViewReviews", "TodoistSearchTasks", "GoogleCalendarGetEventsFromSharedCalendar"],
    "item": ["InventoryManagementSystemGetItemDetails", "GitHubReadIssue", "CiscoUmbrellaGetThreatDetails", "ShopifyGetOrderDetails",
             "FedExShipManagerGetShipmentDetails", "EmergencyDispatchSystemGetIncidentData", "FacebookManagerGetPost",
             "CiscoUmbrellaGetPolicyDetails", "AmazonGetProductDetails", "GitHubGetRepositoryDetails", "GitHubGetUserDetails",
             "ShopifyGetProductDetails", "TwitterManagerGetUserProfile", "TwitterManagerReadTweet", "GoogleCalendarReadEvents",
             "WebBrowserNavigateTo"],
    "message": ["FacebookManagerGetMessages", "SlackSearchMessage", "SlackGetUserDetails", "TeladocSendMessage", "GmailReadEmail",
                "GmailSearchEmails", "TwilioGetReceivedSmsMessages"]}
TYPE = {t: s for s, ts in SURFACE.items() for t in ts}
assert len(TYPE) == 39
SEED = 7579
EVAL_TEST_CASES, EVAL_ROLLOUTS, EVAL_MAX_POS, BENIGN_EVAL_ROLLOUTS = 300, 8, 4, 4


def files():
    for cfg, (_, tag, _) in ARM.items():
        for benign in (False, True):
            for split in ("train", "dev", "test"):
                t = tag + ("_benign" if benign else "") + f"_{split}"
                yield cfg + ("-benign" if benign else ""), split, benign, t


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(cfg, split, benign, tag, rows):
    """Structural checks; returns the per-split summary. Raises on anything that breaks the card's promises."""
    k, n_cases = K[split], (BENIGN_CASES if benign else CASES)[split]
    by = defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    assert len(by) == n_cases and len(rows) == n_cases * k, (tag, len(by), len(rows))
    assert all(sorted(x["rollout"] for x in v) == list(range(k)) for v in by.values()), tag
    assert all(len({(x["tool_response"], x["user_tool"]) for x in v}) == 1 for v in by.values()), "a case's input differs across rollouts"
    assert all(r["violates"] == (r["violates_broad"] if cfg.startswith("realistic") else r["violates_strict"]) for r in rows), tag
    assert all(r.get("benign") in ("neutral", "imperative") for r in rows) if benign else all("benign" not in r for r in rows), tag
    mixed = {c for c, v in by.items() if 0 < sum(x["violates"] for x in v) < k}
    mixed_s = {c for c, v in by.items() if 0 < sum(x["violates_strict"] for x in v) < k}
    mixed_b = {c for c, v in by.items() if 0 < sum(x["violates_broad"] for x in v) < k}
    summ = json.load(open(R / f"{tag}.json"))
    return dict(config=cfg, split=split, cases=len(by), replies=len(rows), k=k,
                strict=sum(r["violates_strict"] for r in rows) / len(rows), broad=sum(r["violates_broad"] for r in rows) / len(rows),
                mixed=len(mixed), mixed_strict=len(mixed_s), mixed_broad=len(mixed_b), mixed_share=len(mixed) / len(by),
                capped=sum(r["finish"] == "length" for r in rows), no_reasoning=sum(not r["reasoning"] for r in rows),
                usd_all=summ["usd"], by_case=by, mixed_set=mixed)


def cuts(rows, tok):
    """Character offset in `reasoning` after its first N tokens (Qwen3.5 tokenizer, reasoning text alone, no special tokens);
    None where the reasoning has fewer than N tokens. N = 0 is offset 0: the input alone."""
    enc = tok([r["reasoning"] for r in rows], add_special_tokens=False, return_offsets_mapping=True)
    out = []
    for r, offs in zip(rows, enc["offset_mapping"]):
        out.append(dict(case=r["case"], rollout=r["rollout"], reasoning_tokens=len(offs),
                        cut={str(n): (0 if n == 0 else offs[n - 1][1] if len(offs) >= n else None) for n in NS}))
    return out


def eval_subset(cfg, split, benign, s):
    """Shared evaluation rows. Injected: mixed cases under the config's label (all on dev; on test a seeded 300 allocated
    across surface x attack in proportion), each with up to 4 violating and the rest non-violating rollouts to 8, drawn
    uniformly within each label, so within-case AUROC is unbiased but rates on the subset are not. Benign: every case,
    4 seeded rollouts."""
    rng = random.Random(f"{SEED}|{cfg}|{split}")
    by = s["by_case"]
    if benign:
        return [dict(case=c, rollout=r) for c in sorted(by) for r in sorted(rng.sample(range(s["k"]), BENIGN_EVAL_ROLLOUTS))]
    cases = sorted(s["mixed_set"])
    if split == "test":
        strata = defaultdict(list)
        for c in cases:
            strata[(TYPE[by[c][0]["user_tool"]], by[c][0]["attack"])].append(c)
        quota = {g: round(EVAL_TEST_CASES * len(v) / len(cases)) for g, v in strata.items()}
        cases = sorted(c for g, v in sorted(strata.items()) for c in rng.sample(sorted(v), min(quota[g], len(v))))
    rows = []
    for c in cases:
        pos = sorted(x["rollout"] for x in by[c] if x["violates"]); neg = sorted(x["rollout"] for x in by[c] if not x["violates"])
        p = rng.sample(pos, min(len(pos), EVAL_MAX_POS))
        n = rng.sample(neg, min(len(neg), EVAL_ROLLOUTS - len(p)))
        p += rng.sample([x for x in pos if x not in p], min(len(pos) - len(p), EVAL_ROLLOUTS - len(p) - len(n)))
        rows += [dict(case=c, rollout=r) for r in sorted(p + n)]
    return rows


def main():
    stage = "--stage" in sys.argv
    present = [(cfg, sp, b, t) for cfg, sp, b, t in files() if (D / f"{t}_rollouts.jsonl").exists()]
    missing = [t for *_, t in files() if not (D / f"{t}_rollouts.jsonl").exists()]
    if missing:
        print("missing:", ", ".join(missing))
        assert not stage, "--stage needs all 12 rollout files"
    tok = None
    if stage:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
        assert tok.is_fast, "offsets need the fast tokenizer"
    S = {}
    for cfg, split, benign, tag in present:
        src = D / f"{tag}_rollouts.jsonl"
        rows = [json.loads(l) for l in open(src)]
        s = S[(cfg, split)] = check(cfg, split, benign, tag, rows)
        s["sha256"] = sha(src)
        if stage:
            for sub, data in [("", None), ("cuts", cuts(rows, tok)), ("eval", eval_subset(cfg, split, benign, s) if split != "train" else None)]:
                p = OUT / sub / cfg / f"{split}.jsonl" if sub else OUT / cfg / f"{split}.jsonl"
                if sub and data is None:
                    continue
                p.parent.mkdir(parents=True, exist_ok=True)
                if not sub:
                    shutil.copyfile(src, p); assert sha(p) == s["sha256"]
                else:
                    p.write_text("".join(json.dumps(x) + "\n" for x in data))
            arm, _, extra = ARM[cfg.removesuffix("-benign")]
            ex = OUT / "prompts" / cfg / f"{split}.jsonl"; ex.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, str(HERE / "e77_variants.py"), "e78v2", arm, "qwen", "--thought", f"--split={split}", "--dry",
                            f"--export={ex}", *extra, *(["--benign"] if benign else [])], check=True, cwd=HERE, capture_output=True)
            keys = {json.loads(l)["case"] for l in open(ex)}
            assert keys == set(s["by_case"]), ("prompts and rollouts disagree on cases", cfg, split)

    print("\n| config | split | cases × K | replies | following strict | following broad | mixed (config label) | mixed share | mixed strict / broad | capped | no reasoning | sha256 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (cfg, split), s in S.items():
        print(f"| {cfg} | {split} | {s['cases']} × {s['k']} | {s['replies']:,} | {s['strict']:.3f} | {s['broad']:.3f} | {s['mixed']} | "
              f"{s['mixed_share']:.2f} | {s['mixed_strict']} / {s['mixed_broad']} | {s['capped']} | {s['no_reasoning']} | `{s['sha256'][:16]}…` |")
    for (cfg, split), s in S.items():
        if cfg in ARM and split == "test":
            by = s["by_case"]; m = s["mixed_set"]
            print(f"\n{cfg} test, mixed by attack:", dict(Counter(by[c][0]["attack"] for c in m)),
                  "| by surface:", {t: f"{sum(TYPE[by[c][0]['user_tool']] == t for c in m)}/{sum(TYPE[v[0]['user_tool']] == t for v in by.values())}" for t in SURFACE})
            tools = sorted({v[0]["user_tool"] for v in by.values()})
            print("| user tool | surface | cases | following | mixed |\n|---|---|---|---|---|")
            for t in tools:
                cs = [c for c, v in by.items() if v[0]["user_tool"] == t]
                print(f"| {t} | {TYPE[t]} | {len(cs)} | {sum(x['violates'] for c in cs for x in by[c]) / (len(cs) * s['k']):.2f} | {sum(c in m for c in cs)} |")
    if stage:
        sums = "".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in sorted(OUT.rglob("*.jsonl")))
        (OUT / "SHA256SUMS").write_text(sums)
        print(f"\nstaged {OUT} ({sum(p.stat().st_size for p in OUT.rglob('*')) / 1e6:.0f} MB); SHA256SUMS written")


if __name__ == "__main__":
    main()
