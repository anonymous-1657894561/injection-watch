"""The one-read guard for InjecForks test, as e79_final.py applies it (decisions/, 2026-10-29 and 10-30): test is read only with
--final, and once. The receipt is written after a run completes, so a run that crashes can be repeated; a second completed read
refuses unless the receipt is removed by hand.

  ap = testread.args(ap)                  # adds --split dev|test (default dev) and --final
  testread.check(a, "e100")               # before anything is loaded
  ...
  testread.done(a, "e100", summary=...)   # after the outputs are written
"""
import json, subprocess, time
from pathlib import Path

AN = Path(__file__).resolve().parent / "results/injecforks/analysis"


def args(ap):
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--final", action="store_true", help="required for --split test")
    return ap


def receipt(name):
    return AN / f"{name}-test-receipt.json"


def check(a, name):
    if a.split != "test":
        return
    if not a.final:
        raise SystemExit("test is read once, for the pre-registered measurement: pass --final")
    if receipt(name).exists():
        raise SystemExit(f"test was already read ({receipt(name)}); a second read is not the pre-registered result")


def done(a, name, **extra):
    if a.split != "test":
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=AN).stdout.strip()
    receipt(name).write_text(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M:%S"), commit=commit, **extra), indent=1))
