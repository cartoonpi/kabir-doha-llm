"""Lesson 6: does the PRD's matra rule set agree with real Kabir dohe?

Runs matra.scan on the cleaned real dohe (val = the most trustworthy text) and reports:
  - how many dohe pass exactly (13+11 on both lines)
  - per half: how far off the counts are (the "error histogram")
  - a few failing dohe, syllable by syllable, so we can see WHICH rule is wrong

Run:  .venv/bin/python lessons/l6_matra.py [val|train|all]
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from matra import scan, syllables  # noqa: E402

which = sys.argv[1] if len(sys.argv) > 1 else "val"
rows = [json.loads(l) for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")]
rows = [r for r in rows if r["split"] != "drop" and (which == "all" or r["split"] == which)]

passed, near, errs = 0, 0, Counter()
fails = []
for r in rows:
    s = scan(r["text"])
    halves = [(x, t) for line in s for x, t in zip(line, (13, 11))]
    d = [None if x is None else x - t for x, t in halves]
    for k, e in enumerate(d):
        errs[("13-half" if k % 2 == 0 else "11-half", e)] += 1
    if all(e == 0 for e in d):
        passed += 1
    else:
        if all(e is not None and abs(e) <= 1 for e in d):
            near += 1
        fails.append((r, s))

print(f"{which}: {len(rows)} dohe")
print(f"  exact 13+11 on both lines: {passed} ({passed / len(rows):.0%})")
print(f"  off by at most 1 in every half: {near} ({near / len(rows):.0%} more)")
for half in ("13-half", "11-half"):
    h = {e: c for (hh, e), c in errs.items() if hh == half}
    total = sum(h.values())
    shown = "  ".join(f"{'none' if e is None else f'{e:+d}'}: {c / total:.0%}"
                      for e, c in sorted(h.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else 0)) if c / total >= 0.01)
    print(f"  {half} error (counted - target): {shown}")

print("\nsome failures, syllable by syllable:")
for r, s in fails[:6]:
    print(f"\n  {r['text'].replace(chr(10), ' / ')}   -> {s}")
    for line in r["text"].split("\n"):
        for half in line.strip(" ।॥").split(","):
            parts = [f"{syl}{w}" for word in half.split() for syl, w in syllables(word)]
            print(f"     {sum(int(p[-1]) for p in parts):2d} = {' '.join(parts)}")
