"""Pick dohe for a human spot check of the Haiku cleanup and write data/cleanup/review_<name>.json.

Takes low/medium-confidence dohe (at most 2/3 of N) plus random high-confidence ones, N in total
(fixed seed), so the check covers both the hard cases and the "easy" ones Haiku was sure of.
For each, a search word for archive.org's search-inside: the longest word of line 1 that
appears verbatim in the 1926 OCR text (the scan's search index is built from that OCR).

Run:  .venv/bin/python data_prep/review_sample.py gurudev 30 [ids to always include, e.g. 0,64,125]
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def main(name, n, must=()):
    d = ROOT / "data/cleanup"
    inp = {r["i"]: r for r in map(json.loads, (d / f"pilot_input_{name}.jsonl").open(encoding="utf-8"))}
    out = [json.loads(l) for l in (d / f"pilot_output_{name}.jsonl").open(encoding="utf-8")]
    ocr = (ROOT / "data/raw/belvedere_1926.txt").read_text(encoding="utf-8")

    hard = [r for r in out if r["confidence"] != "high"]
    easy = [r for r in out if r["confidence"] == "high"]
    random.seed(0)
    forced = [r for r in out if r["i"] in must]
    hard = [r for r in hard if r["i"] not in must]
    easy = [r for r in easy if r["i"] not in must]
    hard = random.sample(hard, max(0, min(len(hard), (2 * n) // 3 - len(forced))))   # keep at least a third "easy"
    pick = forced + hard + random.sample(easy, max(0, min(len(easy), n - len(forced) - len(hard))))
    pick.sort(key=lambda r: r["i"])

    rows = []
    for r in pick:
        src = inp[r["i"]]
        words = sorted(src["copy_1926"].split("\n")[0].replace(",", " ").replace("।", " ").split(), key=len, reverse=True)
        word = next((w for w in words if len(w) >= 3 and w in ocr), words[0] if words else "")
        rows.append({**r, "num": src.get("num"), "copy_1926": src["copy_1926"], "copy_1912": src.get("copy_1912"),
                     "search_word": word})
    (d / f"review_{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"review_{name}.json: {len(rows)} dohe ({sum(r['confidence'] != 'high' for r in rows)} low/medium, "
          f"{sum(r['confidence'] == 'high' for r in rows)} high)")


if __name__ == "__main__":
    must = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else set()
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30, must)
