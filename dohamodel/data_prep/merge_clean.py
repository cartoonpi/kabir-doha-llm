"""Merge the Haiku cleanup outputs, run automatic checks, and split train / val.

Inputs : data/interim/aligned.jsonl (section, both OCR copies)
         data/cleanup/pilot_output_gurudev.jsonl + data/cleanup/batch_*_output.jsonl (Haiku)
Output : data/clean/kabir_real.jsonl, one doha per line:
    {"id", "section", "text", "basis", "confidence", "flags": [...], "split": "train"|"val"|"drop"}
         data/clean/report.md (counts, flagged examples)

Checks (each adds a flag):
  format      text is not "line1।\\nline2॥"
  fragment    a line has fewer than 3 words or 12 characters (a broken half-line)
  replaced    a word changed by 3+ letters vs. the OCR word in the SAME position of the line
              (both copies checked; aligned word-by-word, unlike the pilot's first check)
  duplicate   the same doha appeared earlier in the book (popular dohe are repeated)
Split: "drop" = format/fragment/duplicate. "val" = 15% of all kept dohe, drawn only from
two-copy, high-confidence, unflagged dohe (the most trustworthy text). Everything else kept = train.

Run:  .venv/bin/python data_prep/merge_clean.py
"""
import difflib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
VAL_SHARE = 0.15
# ids that are not dohe: the publisher's advertisements after the last section, which
# extract.py's pairing let through (spotted by the Haiku cleanup agent for batch 08)
NOT_DOHE = set(range(2059, 2080))
WORD = re.compile(r"[ऀ-ॣ०-ॿ]+")


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def replaced_words(new, old):
    """Words of `new` that replace a word of `old` at the same position by 3+ letters."""
    out = []
    for nl, ol in zip(new.split("\n"), old.split("\n")):
        a, b = WORD.findall(ol), WORD.findall(nl)
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "replace":
                for x, y in zip(a[i1:i2], b[j1:j2]):
                    if lev(x, y) >= 3:
                        out.append((x, y))
            elif op == "insert":
                out += [("", y) for y in b[j1:j2]]
    return out


O, CONFUSED = "\u094b", ("\u093e", "\u0947")   # ो, and the two signs OCR turns it into (ा, े)


def rhyme_fix(text, freq):
    """The two lines of a doha rhyme. If one line ends in -ोX and the other in -ाX or -ेX
    (होय / हाय, कोय / काय), the second is the old-type OCR error: ो printed as े + ा, half lost.
    Only if the fixed word is more common in the book than the word it replaces, so a real,
    frequent word (जाय) is never "fixed" into a rare one (जोय).
    Returns (new_text, list of 'old→new')."""
    l1, l2 = text.split("\n")
    w = [re.findall(r"[\u0900-\u097F]+", l)[-1] if re.findall(r"[\u0900-\u097F]+", l) else "" for l in (l1.rstrip("।"), l2.rstrip("॥"))]
    if len(w[0]) < 2 or len(w[1]) < 2:
        return text, []
    for good, bad, k in ((0, 1, 1), (1, 0, 0)):
        g, b = w[good], w[bad]
        # same final letter, the vowel sign before it: ो in one line, ा/े in the other
        if g[-1] == b[-1] and g[-2] == O and b[-2] in CONFUSED:
            fixed = b[:-2] + O + b[-1]
            if freq[fixed] <= freq[b]:
                continue
            lines = [l1, l2]
            i = lines[k].rfind(b)
            lines[k] = lines[k][:i] + fixed + lines[k][i + len(b):]
            return "\n".join(lines), [f"{b}→{fixed}"]
    return text, []


def main():
    aligned = {r["i"]: r for r in map(json.loads, (ROOT / "data/interim/aligned.jsonl").open(encoding="utf-8"))}
    outs = [ROOT / "data/cleanup/pilot_output_gurudev.jsonl", *sorted((ROOT / "data/cleanup").glob("batch_*_output.jsonl"))]
    haiku = {}
    for f in outs:
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            haiku[r["i"]] = r
    missing = [i for i in aligned if i not in haiku]
    print(f"Haiku rows: {len(haiku)} of {len(aligned)}   missing: {len(missing)}")

    freq = Counter(w for h in haiku.values() for w in WORD.findall(h["text"]))
    rows, seen = [], {}
    for i, a in sorted(aligned.items()):
        if i not in haiku:
            continue
        h = haiku[i]
        text = h["text"].strip()
        flags = ["not_doha"] if i in NOT_DOHE else []
        if "\n" in text:
            text, fixes = rhyme_fix(text, freq)
            flags += [f"rhyme_fix:{f}" for f in fixes]
        if not re.fullmatch(r"[^\n।॥]+।\n[^\n।॥]+॥", text):
            flags.append("format")
        lines = [l.strip(" ।॥,") for l in text.split("\n")]
        if any(len(WORD.findall(l)) < 3 or len(l) < 12 for l in lines) or len(lines) != 2:
            flags.append("fragment")
        # a replacement counts only if it is far from BOTH copies at that position
        r26 = {y for _, y in replaced_words(text, a["text_1926"])}
        r12 = {y for _, y in replaced_words(text, a["text_1912"])} if a.get("text_1912") and a["kind"] != "doubtful" else r26
        if r26 & r12:
            flags.append("replaced:" + " ".join(sorted(r26 & r12)))
        key = re.sub(r"[^ऀ-ॿ]", "", text)
        if key in seen:
            flags.append(f"duplicate:{seen[key]}")
        else:
            seen[key] = i
        rows.append({"id": i, "section": a["section"], "text": text, "basis": h.get("basis"),
                     "confidence": h.get("confidence"), "flags": flags, "note": h.get("note", "")})

    for r in rows:
        r["split"] = "drop" if any(f.split(":")[0] in ("not_doha", "format", "fragment", "duplicate") for f in r["flags"]) else "train"
    kept = [r for r in rows if r["split"] == "train"]
    pool = [r for r in kept if r["basis"] == "both" and r["confidence"] == "high" and not r["flags"]]
    random.seed(0)
    for r in random.sample(pool, min(len(pool), round(VAL_SHARE * len(kept)))):
        r["split"] = "val"

    out = ROOT / "data/clean/kabir_real.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    split, conf = Counter(r["split"] for r in rows), Counter(r["confidence"] for r in rows)
    flags = Counter(f.split(":")[0] for r in rows for f in r["flags"])
    chars = sum(len(r["text"]) + 2 for r in rows if r["split"] != "drop")
    lines = [f"# Real Kabir dohe: cleanup report", "",
             f"- dohe: {len(rows)} (Haiku missing: {len(missing)})",
             f"- split: {dict(split)}  · characters kept: {chars:,}",
             f"- confidence: {dict(conf)}",
             f"- flags: {dict(flags)}",
             f"- val pool (two-copy, high, unflagged): {len(pool)}", "",
             "## Flagged examples", ""]
    for kind in ("fragment", "replaced", "duplicate", "format"):
        ex = [r for r in rows if any(f.startswith(kind) for f in r["flags"])][:8]
        lines.append(f"### {kind} ({flags[kind]})")
        lines += [f"- #{r['id']} [{r['confidence']}] {r['text'].replace(chr(10), ' / ')}  `{' '.join(r['flags'])}`" for r in ex]
        lines.append("")
    (ROOT / "data/clean/report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
