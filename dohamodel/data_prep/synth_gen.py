"""Full synthetic data set (PRD build step 4): waves of Haiku agents, then one filter over everything.

Same tags, prompt and fitness function as the pilot (synth_pilot.py), with three changes from its lessons:
  - 100 requests per agent (the pilot's 20 left most of the tokens in per-agent overhead)
  - themes are weighted by 1 / pilot keep rate, so every theme ends with about as many kept dohe
  - the prompt says what each image means in Kabir (the pilot used the heron for patience)

  make N     writes data/synth/full/wave_NN/agent_XX/{requests.jsonl, prompt.md}
  score      filters every wave's outputs -> data/synth/full/{scored.jsonl, kept.jsonl}
             format -> meter (one half may be off by 1; kept rows are graded exact or near) -> copy (5-word run from a real doha) -> near-duplicate of an earlier kept doha
             (character 4-gram Jaccard >= 0.5, via an inverted index so it scales to 50K dohe)

Run:  .venv/bin/python data_prep/synth_gen.py make 1 [--agents 10 --per 100]
      .venv/bin/python data_prep/synth_gen.py score
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "data_prep"))
sys.path.insert(0, str(ROOT))
from synth_pilot import ALLOWED, DEVICES, GRAMS, IMAGES, PROMPT, SHAPE, THEMES, examples_pool, halves, rows  # noqa: E402
from matra import check, grade  # noqa: E402

OUT = ROOT / "data/synth/full"

# pilot keep rate per theme (data/synth/pilot, 2026-10-28)
PILOT_KEEP = {"ahankar": 6 / 17, "bhakti": 8 / 19, "dhairya": 7 / 20, "guru": 4 / 18, "maya": 7 / 17,
              "mrityu": 15 / 20, "pakhand": 10 / 18, "prem": 10 / 23, "satsang": 14 / 22, "vani": 14 / 26}

# what each image stands for in Kabir, so Haiku uses it with the right meaning
MEANING = {
    "deepak": "inner light of knowledge; the lamp lit within drives out darkness",
    "kumhar": "the guru as potter who shapes the pot, striking outside while supporting inside; or the clay that one day kneads the potter (death)",
    "kuan": "a deep well; thirst and drawing water; the true water lies within",
    "machhli": "a fish that dies out of water: love and longing; a fish thirsty in water (paradox)",
    "mala": "prayer beads turned while the mind does not turn: empty ritual",
    "chandan": "sandalwood keeps its fragrance though snakes coil round it: good character unspoiled by bad company",
    "patang": "the moth that burns itself in the flame: love's self-sacrifice, or ruin by craving",
    "bazaar": "the world as a marketplace where one stands and trades in deeds",
    "naav": "the boat that crosses the ocean of existence; the guru as boatman; a leaky boat",
    "vriksh": "a tree that gives fruit and shade to others, not to itself; a tall date-palm gives no shade",
    "kargha": "the weaver's loom (Kabir was a weaver): life and the body woven thread by thread",
    "charkha": "the spinning wheel: breath, steady remembrance, spinning the thread of life",
    "kapas": "raw cotton that becomes cloth only after being beaten and spun: suffering refines",
    "chadar": "the sheet of life, to be worn with care and returned unstained",
    "dhaga": "the thread of love: once broken, it joins only with a knot",
    "sui": "the needle that joins (unlike scissors that cut); small yet useful",
    "chakki": "the grinding mill whose two stones crush the world; only the grain near the axle survives",
    "kolhu": "the blindfolded ox circling the oil press: the endless worldly round",
    "bhatti": "the furnace or kiln: trials that refine gold; the still that distils the wine of love",
    "kasauti": "the touchstone that tests true gold: testing a true devotee",
    "tarazu": "the scales that weigh true worth; deeds weighed",
    "rassi": "the well rope; a soft rope that wears a groove in stone with practice",
    "ghada": "the clay pot of the body that breaks; water in a pot within water (the soul in God)",
    "koyal": "the cuckoo whose sweet voice wins hearts (looks like the crow until it speaks)",
    "hans": "the swan of discernment that picks pearls and separates milk from water; the soul",
    "kauwa": "the crow: harsh speech and impurity; crow and cuckoo look alike until they speak",
    "kasturi": "the musk deer searching the forest for the scent in its own navel: God is within",
    "bagula": "the heron standing still as if in meditation while eyeing fish: the hypocrite",
    "chakor": "the chakor bird gazing only at the moon: single-minded devoted longing",
    "papiha": "the papiha bird that drinks only the swati raindrop: single-minded thirst for God",
}
assert set(MEANING) == {i["id"] for i in IMAGES}

EXTRA = """
What the images mean in Kabir (use them with this meaning):
{meanings}

You have {n} requests. Keep every doha different: vary the opening words, the rhymes and the pictures.
Write the output in two Write calls so no single file is too long: requests 1-{h} to {out_a}, the rest to {out_b}.
Each file: one JSON object per line, {{"rid": "<rid exactly as given, a quoted string>", "text": "<line 1>।\\n<line 2>॥"}}."""

# self-check variant: the agent may run the meter checker on its own drafts and fix the failures
SELFCHECK = """
What the images mean in Kabir (use them with this meaning):
{meanings}

You have {n} requests. Keep every doha different: vary the opening words, the rhymes and the pictures.
Put the only comma of each line at the caesura (write "साधो मन ..." not "साधो, मन ...").

Work in rounds:
1. Write all {n} dohe with the Write tool to {out}: one JSON object per line,
   {{"rid": "<rid exactly as given, a quoted string>", "text": "<line 1>।\\n<line 2>॥"}}
2. Check the meter by running exactly this command with the Bash tool:
   {cli} {out}
   It prints PASS or FAIL per doha, the matra count of each half, and for a FAIL which half is off and by how much
   ("line 1 first half is +2 (remove 2)"). A doha passes when every half is 13 / 11 / 13 / 11, or only one half is off by 1.
3. Rewrite ONLY the failing dohe (keep their rid, theme, images and device; change words to fix the count, e.g. a long
   vowel for a short one, or drop or add a short word). Keep the passing ones unchanged. Write the whole file again.
4. Repeat steps 2-3 until all pass or you have done 3 rounds of fixes. Do not run any other command, and do not
   create any file other than {out} (use the Write tool for it; split nothing into helper files).
Reply with just "done", the number of lines, and the final "N/M pass" line from the checker."""


def make(wave, agents, per, selfcheck=False):
    rng = random.Random(1000 + wave)
    # without self-check, rarely-kept themes get more requests; with it, every theme passes ~95%, so equal weights
    weights = [1.0 if selfcheck else 1 / PILOT_KEEP[t["id"]] for t in THEMES]
    for a in range(agents):
        d = OUT / f"wave_{wave:02d}/agent_{a:02d}"
        d.mkdir(parents=True, exist_ok=True)
        reqs = []
        for k in range(per):
            t = rng.choices(THEMES, weights)[0]
            v = rng.choice(DEVICES)
            imgs = rng.sample(IMAGES, rng.choice([1, 2]))
            reqs.append({
                "rid": f"w{wave:02d}a{a:02d}r{k:03d}",
                "theme": t["id"], "theme_desc": f"{t['name']}: {t['desc']}",
                "images": [{"id": i["id"], "name": i["name"]} for i in imgs],
                "device": v["id"], "device_desc": v["name"],
            })
        (d / "requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs), encoding="utf-8")
        ex = "\n\n".join(rng.sample(examples_pool, 6))
        base = PROMPT.format(examples=ex, req_path=d / "requests.jsonl", out_path="(see below)")
        # the pilot prompt's output paragraph is replaced by the two-file instruction
        base = base[:base.index("Write the result with the Write tool")].rstrip()
        used = sorted({i["id"] for r in reqs for i in r["images"]})
        meanings = "\n".join(f"- {i['name']} ({i['id']}): {MEANING[i['id']]}" for i in IMAGES if i["id"] in used)
        if selfcheck:
            base = base.replace("Do NOT run code or count matras with a script; just write as a poet would. ", "")
            cli = f"{ROOT / '.venv/bin/python'} {ROOT / 'data_prep/meter_cli.py'}"
            text = base + "\n" + SELFCHECK.format(meanings=meanings, n=per, out=d / "output_a.jsonl", cli=cli) + "\n"
        else:
            text = base + "\n" + EXTRA.format(meanings=meanings, n=per, h=per // 2, out_a=d / "output_a.jsonl",
                                              out_b=d / "output_b.jsonl") + "\nReply with just \"done\" and the number of lines written.\n"
        (d / "prompt.md").write_text(text, encoding="utf-8")
    print(f"wave {wave}: {agents} agents x {per} requests in {OUT / f'wave_{wave:02d}'}")


VOCATIVE = re.compile(r"(?m)^((?:साधो|संतो|भाई|रे मन|मन रे|सुनो भाई|सुनि भाई|साधु|रे)),\s*")


def shingles(text, k=4):
    s = re.sub(r"[\s,।॥]+", " ", text)
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def score():
    reqs, outs = {}, {}
    for d in sorted(OUT.glob("wave_*/agent_*")):
        for l in (d / "requests.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            reqs[r["rid"]] = r
        for f in sorted(d.glob("output_*.jsonl")):
            for l in f.open(encoding="utf-8"):
                if l.strip():
                    try:
                        o = json.loads(l)
                        outs[str(o["rid"])] = o["text"].strip()
                    except (json.JSONDecodeError, KeyError):
                        pass
    index, kept_sh, results = defaultdict(list), [], []
    for rid, r in sorted(reqs.items()):
        text = outs.get(rid)
        if text:
            # a comma after a line-opening vocative (साधो, रे मन,) is address, not the caesura
            text = VOCATIVE.sub(r"\1 ", text)
        if text is None:
            verdict = "missing"
        elif not (SHAPE.match(text) and ALLOWED.match(text)):
            verdict = "format"
        elif not check(text, nasal=True, flex=True, slack=1):
            verdict = "meter"
        elif any(tuple(w[i:i + 5]) in GRAMS for w in [re.findall(r"\S+", text.replace("\n", " "))] for i in range(len(w) - 4)):
            verdict = "copy"
        else:
            sh = shingles(text)
            cand = Counter(j for g in sh for j in index[g])
            dup = any(len(sh & kept_sh[j]) / len(sh | kept_sh[j]) >= 0.5 for j, _ in cand.most_common(20))
            verdict = "duplicate" if dup else "kept"
            if not dup:
                for g in sh:
                    index[g].append(len(kept_sh))
                kept_sh.append(sh)
        results.append({**r, "text": text, "verdict": verdict, "meter": grade(text) if text else None,
                        "halves": halves(text) if verdict not in ("missing", "format") else None})
    (OUT / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")
    kept = [{"rid": r["rid"], "theme": r["theme"], "images": [i["id"] for i in r["images"]], "device": r["device"],
             "meter": r["meter"], "text": r["text"]} for r in results if r["verdict"] == "kept"]
    (OUT / "kept.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8")

    c, n = Counter(r["verdict"] for r in results), len(results)
    print(f"{n} requests: " + "   ".join(f"{k} {v}" for k, v in c.most_common()) + f"   -> kept {c['kept'] / n:.0%}")
    print(f"kept characters: {sum(len(k['text']) for k in kept):,}")
    by_wave = defaultdict(Counter)
    for r in results:
        by_wave[r["rid"][:3]][r["verdict"]] += 1
    for w, cw in sorted(by_wave.items()):
        ex = sum(r["verdict"] == "kept" and r["meter"] == "exact" for r in results if r["rid"][:3] == w)
        print(f"  {w}: kept {cw['kept']}/{sum(cw.values())} ({cw['kept'] / sum(cw.values()):.0%}; exact {ex}, near {cw['kept'] - ex})"
              f"   meter {cw['meter']}   format {cw['format']}   missing {cw['missing']}")
    for key in ("theme", "device"):
        print(f"kept by {key}: " + "  ".join(f"{k} {v}" for k, v in sorted(Counter(k[key] for k in kept).items())))
    img = Counter(i for k in kept for i in k["images"])
    print(f"kept per image: min {min(img.values()) if img else 0}  max {max(img.values()) if img else 0}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    p.add_argument("wave", nargs="?", type=int)
    p.add_argument("--agents", type=int, default=10)
    p.add_argument("--per", type=int, default=100)
    p.add_argument("--selfcheck", action="store_true")
    a = p.parse_args()
    make(a.wave, a.agents, a.per, a.selfcheck) if a.cmd == "make" else score()
