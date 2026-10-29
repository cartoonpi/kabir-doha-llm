"""Meter report for a JSONL file of dohe ({"rid": ..., "text": ...} per line), for the self-checking Haiku agents.

Prints one line per doha: PASS (exact or near) or FAIL, with each half's matra count and what to change.
A doha passes if all four halves scan 13 / 11 / 13 / 11, or if just one half is off by 1 (near).

Run:  .venv/bin/python data_prep/meter_cli.py <file.jsonl>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from matra import count_range, grade  # noqa: E402

NAMES = ["line 1 first half", "line 1 second half", "line 2 first half", "line 2 second half"]
TARGETS = [13, 11, 13, 11]


def report(text):
    lines = text.strip().split("\n")
    if len(lines) != 2 or any("," not in l for l in lines):
        return "FAIL  shape: need 2 lines, each with one comma at the caesura"
    halves = [h for l in lines for h in l.strip(" ।॥").split(",", 1)]
    g = grade(text)
    parts, fixes = [], []
    for name, h, t in zip(NAMES, halves, TARGETS):
        lo, hi = count_range(h, nasal=True, flex=True)
        parts.append(f"{lo}" if lo == hi else f"{lo}-{hi}")
        if not lo <= t <= hi:
            d = lo - t if lo > t else hi - t
            fixes.append(f"{name} is {d:+d} (remove {d})" if d > 0 else f"{name} is {d:+d} (add {-d})")
    counts = " + ".join(parts[:2]) + " / " + " + ".join(parts[2:])
    if g:
        return f"PASS  {g:5s}  {counts}"
    return f"FAIL  {counts}   " + "; ".join(fixes)


if __name__ == "__main__":
    n = ok = 0
    for l in Path(sys.argv[1]).open(encoding="utf-8"):
        if not l.strip():
            continue
        try:
            o = json.loads(l)
        except json.JSONDecodeError:
            print("FAIL  bad JSON line:", l.strip()[:60])
            continue
        r = report(o["text"])
        n += 1
        ok += r.startswith("PASS")
        print(f"{o['rid']}  {r}")
    print(f"\n{ok}/{n} pass")
