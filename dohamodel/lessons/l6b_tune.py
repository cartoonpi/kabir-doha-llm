"""Lesson 6b: tune the matra checker, measuring BOTH kinds of error.

  pass rate on real val dohe   = how often it accepts good dohe     (want high)
  pass rate on broken dohe     = how often it accepts bad dohe      (want low)

Broken dohe are real val dohe with one random word deleted, or one random word repeated,
which shifts a half by 1-4 matras: exactly the kind of near-miss the model will produce.

Run:  .venv/bin/python lessons/l6b_tune.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from matra import check  # noqa: E402

rows = [json.loads(l) for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")]
val = [r["text"] for r in rows if r["split"] == "val"]


def break_doha(text, rng, mode):
    """Delete or repeat one word in one random half (keeping the comma and line marks)."""
    lines = text.split("\n")
    li = rng.randrange(2)
    body, end = lines[li][:-1], lines[li][-1]          # strip । or ॥
    halves = body.split(", ")
    hi = rng.randrange(len(halves))
    words = halves[hi].split()
    if len(words) < 2:
        return None
    k = rng.randrange(len(words))
    words = words[:k] + words[k + 1:] if mode == "delete" else words[:k + 1] + words[k:]
    halves[hi] = " ".join(words)
    lines[li] = ", ".join(halves) + end
    return "\n".join(lines)


rng = random.Random(0)
broken = [b for t in val for mode in ("delete", "repeat") if (b := break_doha(t, rng, mode))]

print(f"{len(val)} real val dohe, {len(broken)} broken copies\n")
print(f"{'checker':28s} {'accepts real':>13s} {'accepts broken':>15s}")
for name, kw in [("PRD rules", {}), ("A: nasal rule", {"nasal": True}),
                 ("B: flexible syllables", {"flex": True}), ("A + B", {"nasal": True, "flex": True})]:
    good = sum(check(t, **kw) for t in val) / len(val)
    bad = sum(check(t, **kw) for t in broken) / len(broken)
    print(f"{name:28s} {good:13.0%} {bad:15.0%}")
