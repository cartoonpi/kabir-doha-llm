"""Step 3 of real data: line up the 1926 and 1912 printings doha by doha.

Both are the same book, so their dohe come in the same order, but each extraction lost a
few dohe and OCR'd the rest differently. We find the best one-to-one matching that keeps
the order: classic sequence alignment (the same dynamic programming used for DNA and diff).

    score(i, j) = best alignment of the first i dohe of 1926 with the first j of 1912
               = max( score(i-1, j-1) + match(i, j),     # pair them
                      score(i-1, j)   - GAP,             # 1926 doha has no partner
                      score(i,   j-1) - GAP )            # 1912 doha has no partner

match = similarity of the two texts (0..1) minus 0.5, so only real look-alikes pay to pair.
We only fill cells near the diagonal (a "band"), since both lists are about the same length.

Output: data/interim/aligned.jsonl (one row per 1926 doha, with its 1912 partner if any)
Run:  .venv/bin/python data_prep/align.py
"""
import difflib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
GAP, BAND = 0.2, 150


def load(year):
    return [json.loads(l) for l in (ROOT / f"data/interim/belvedere_{year}.jsonl").open(encoding="utf-8")]


def key(d):
    return d["text"].replace(" ", "").replace(",", "").replace("।", "").replace("॥", "").replace("\n", "")


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def align(A, B):
    ka, kb = [key(d) for d in A], [key(d) for d in B]
    n, m = len(A), len(B)
    NEG = float("-inf")
    score = {(0, 0): 0.0}
    back = {}
    for i in range(n + 1):
        centre = round(i * m / max(n, 1))
        for j in range(max(0, centre - BAND), min(m, centre + BAND) + 1):
            if i == j == 0:
                continue
            opts = []
            if i and j and (i - 1, j - 1) in score:
                opts.append((score[i - 1, j - 1] + sim(ka[i - 1], kb[j - 1]) - 0.5, "pair"))
            if i and (i - 1, j) in score:
                opts.append((score[i - 1, j] - GAP, "skipA"))
            if j and (i, j - 1) in score:
                opts.append((score[i, j - 1] - GAP, "skipB"))
            if opts:
                score[i, j], back[i, j] = max(opts)
    # walk back from the end to recover the matching
    i, j, pairs = n, m, {}
    while i or j:
        move = back[i, j]
        if move == "pair":
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == "skipA":
            i -= 1
        else:
            j -= 1
    return pairs, ka, kb


def char_diffs(a, b):
    """Where two texts disagree, as (1926 piece, 1912 piece) pairs."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def main():
    A, B = load(1926), load(1912)
    pairs, ka, kb = align(A, B)
    rows, kinds = [], Counter()
    agree_chars = total_chars = 0
    for i, a in enumerate(A):
        j = pairs.get(i)
        row = {"i": i, "section": a["section"], "num": a["num"], "text_1926": a["text"]}
        if j is None:
            kind = "unmatched"
        else:
            b = B[j]
            s = sim(ka[i], kb[j])
            kind = "identical" if s == 1 else "close" if s >= 0.85 else "loose" if s >= 0.6 else "doubtful"
            row.update(j=j, text_1912=b["text"], sim=round(s, 3), diffs=char_diffs(a["text"], b["text"]))
            sm = difflib.SequenceMatcher(None, ka[i], kb[j], autojunk=False)
            agree_chars += sum(t.size for t in sm.get_matching_blocks())
            total_chars += max(len(ka[i]), len(kb[j]))
        row["kind"] = kind
        kinds[kind] += 1
        rows.append(row)

    out = ROOT / "data/interim/aligned.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    print(f"1926: {len(A)} dohe   1912: {len(B)} dohe   paired: {len(pairs)}")
    for k in ("identical", "close", "loose", "doubtful", "unmatched"):
        print(f"  {k:10s} {kinds[k]:5d}  ({kinds[k] / len(A):.0%})")
    print(f"character agreement inside paired dohe: {agree_chars / max(total_chars, 1):.1%}")
    diffs = Counter(d for r in rows for d in map(tuple, r.get("diffs", [])))
    print("most common disagreements (1926 vs 1912):")
    for (x, y), c in diffs.most_common(20):
        print(f"   {x!r:8} vs {y!r:8} x{c}")


if __name__ == "__main__":
    main()
