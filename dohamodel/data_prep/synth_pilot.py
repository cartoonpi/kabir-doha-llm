"""Synthetic data pilot (PRD build step 3): 200 Haiku dohe from random tag combinations.

  make   writes data/synth/pilot/agent_XX/{requests.jsonl, prompt.md}: 10 agents x 20 requests.
         Each request = 1 theme + 1-2 images + 1 device, and the prompt shows 4 real Kabir dohe
         (train split, high confidence, scanning 13+11) so the language stays close to Kabir's.
  score  reads every agent_XX/output.jsonl and applies the PRD's fitness function:
         format -> meter (matra.check, nasal+flex) -> copy (5-word run from a real doha)
         -> duplicate (same text, or a near-duplicate of an earlier kept one).

Tag ids come from the classifier's config/tags.yaml (the original 10 themes, the first 30 images);
the 4 devices are from the PRD and are not in tags.yaml yet.

Run:  .venv/bin/python data_prep/synth_pilot.py make|score
"""
import difflib
import json
import random
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from matra import check, count_range  # noqa: E402

TAGS = yaml.safe_load((ROOT.parent / "clasifierLM/classifier/config/tags.yaml").open(encoding="utf-8"))
V1_THEMES = ["guru", "maya", "mrityu", "ahankar", "pakhand", "bhakti", "vani", "dhairya", "satsang", "prem"]
THEMES = [t for t in TAGS["themes"] if t["id"] in V1_THEMES]
IMAGES = TAGS["images"][:30]
DEVICES = [
    {"id": "ulatbansi", "name": "paradox (ulatbansi): an upside-down image that seems impossible, e.g. the ocean caught fire"},
    {"id": "sadho", "name": "direct address: speaks to the listener, e.g. साधो, सुनो भाई, रे मन"},
    {"id": "prashn", "name": "rhetorical question: asks a question the listener must answer for themselves"},
    {"id": "kahekabir", "name": "signature line: names the poet, e.g. कहै कबीर or कबीरा"},
]
N_AGENTS, PER_AGENT = 10, 20
OUT = ROOT / "data/synth/pilot"

rows = [json.loads(l) for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")]
real = [r["text"] for r in rows if r["split"] != "drop"]
examples_pool = [r["text"] for r in rows
                 if r["split"] == "train" and r["confidence"] == "high" and check(r["text"], nasal=True, flex=True)]

PROMPT = """You are writing NEW Kabir-style dohe (Sant Kabir's 15th-century Sadhukkadi Hindi, in Devanagari) as
training data for a small poetry model. Kabir's language: simple rural words, old forms (कोय, होय, जाय,
माहिं, नाहिं, सों, तें, बिनु, कहै, जिन, पाया), concrete everyday images, a sharp moral turn. NOT modern Hindi.

The form, exactly:
  line 1: first half, second half।
  line 2: first half, second half॥
- a comma at the caesura of each line; line 1 ends with । and line 2 ends with ॥
- each line is 13 matras + 11 matras (short syllable = 1, long = 2; a short syllable before a joined
  consonant or with ं is 2; ँ adds nothing)
- the two lines rhyme on their last word

Four real Kabir dohe (style reference only; never reuse their lines):
{examples}

Write ONE new doha for EACH request in {req_path}. Each request gives a theme, one or two images and a
device. The doha must be about the theme, use each image as a concrete picture, and use the device.
Do not copy any real Kabir doha you know, and make every doha different from the others.

Do NOT run code or count matras with a script; just write as a poet would. Use only Devanagari, spaces,
the comma, । and ॥ inside the doha.

Write the result with the Write tool to {out_path}: one JSON object per line, exactly
  {{"rid": <request rid>, "text": "<line 1>।\\n<line 2>॥"}}
(the two lines joined by a real newline escaped as \\n in JSON). 20 lines, one per request, same order.
Reply with just "done" and the number of lines written."""


def make():
    rng = random.Random(0)
    for a in range(N_AGENTS):
        d = OUT / f"agent_{a:02d}"
        d.mkdir(parents=True, exist_ok=True)
        reqs = []
        for k in range(PER_AGENT):
            reqs.append({
                "rid": a * PER_AGENT + k,
                "theme": (t := rng.choice(THEMES))["id"], "theme_desc": f"{t['name']}: {t['desc']}",
                "images": [{"id": i["id"], "name": i["name"]} for i in rng.sample(IMAGES, rng.choice([1, 2]))],
                "device": (v := rng.choice(DEVICES))["id"], "device_desc": v["name"],
            })
        (d / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs), encoding="utf-8")
        ex = "\n\n".join(rng.sample(examples_pool, 4))
        (d / "prompt.md").write_text(PROMPT.format(examples=ex, req_path=d / "requests.jsonl", out_path=d / "output.jsonl"),
                                     encoding="utf-8")
    print(f"{N_AGENTS} agents x {PER_AGENT} requests in {OUT}   (example pool: {len(examples_pool)} real dohe)")


real_words = [re.findall(r"\S+", d.replace("\n", " ")) for d in real]
GRAMS = {tuple(w[i:i + 5]) for w in real_words for i in range(len(w) - 4)}
SHAPE = re.compile(r"^[^\n,]+, [^\n,]+।\n[^\n,]+, [^\n,]+॥$")
ALLOWED = re.compile(r"^[ऀ-ॿ ,\n]+$")


def halves(text):
    out = []
    for line in text.split("\n"):
        for h in line.strip(" ।॥").split(",", 1):
            out.append(count_range(h, nasal=True, flex=True))
    return out


def score():
    reqs, outs = {}, {}
    for d in sorted(OUT.glob("agent_*")):
        for l in (d / "requests.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            reqs[r["rid"]] = r
        f = d / "output.jsonl"
        if f.exists():
            for l in f.open(encoding="utf-8"):
                if l.strip():
                    o = json.loads(l)
                    outs[o["rid"]] = o["text"].strip()
    results, kept = [], []
    for rid, r in sorted(reqs.items()):
        text = outs.get(rid)
        if text is None:
            verdict = "missing"
        elif not (SHAPE.match(text) and ALLOWED.match(text)):
            verdict = "format"
        elif not check(text, nasal=True, flex=True):
            verdict = "meter"
        elif any(tuple(w[i:i + 5]) in GRAMS for w in [re.findall(r"\S+", text.replace("\n", " "))] for i in range(len(w) - 4)):
            verdict = "copy"
        elif any(difflib.SequenceMatcher(None, text, k).ratio() > 0.75 for k in kept):
            verdict = "duplicate"
        else:
            verdict = "kept"
            kept.append(text)
        results.append({**r, "text": text, "verdict": verdict,
                        "halves": halves(text) if verdict not in ("missing",) and text else None})
    (OUT / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    from collections import Counter
    c = Counter(r["verdict"] for r in results)
    n = len(results)
    print(f"{n} requests: " + "   ".join(f"{k} {v}" for k, v in c.most_common()))
    written = [r for r in results if r["verdict"] not in ("missing", "format")]
    print(f"rhythm pass among well-formed: {sum(r['verdict'] not in ('meter',) for r in written)}/{len(written)}"
          f"   kept overall: {c['kept']}/{n} = {c['kept'] / n:.0%}")
    # per-half: how close is Haiku to 13 / 11?
    for pos, target in [(0, 13), (1, 11)]:
        hs = [r["halves"][j] for r in written for j in (pos, pos + 2) if len(r["halves"]) == 4]
        exact = sum(lo <= target <= hi for lo, hi in hs) / len(hs)
        near = sum(lo - 1 <= target <= hi + 1 for lo, hi in hs) / len(hs)
        print(f"  {target}-halves: exact {exact:.0%}   within ±1 {near:.0%}")
    real_val = [r["text"] for r in rows if r["split"] == "val"]
    print(f"  (real val dohe passing the same check: {sum(check(t, nasal=True, flex=True) for t in real_val) / len(real_val):.0%})")
    for key in ("theme", "device"):
        by = Counter(r[key] for r in results)
        ok = Counter(r[key] for r in results if r["verdict"] == "kept")
        print(f"kept by {key}: " + "  ".join(f"{k} {ok[k]}/{by[k]}" for k in sorted(by)))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
