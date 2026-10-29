"""Validate data/requests_human.jsonl against config/tags.yaml.

Run from classifier/:   .venv/bin/python -m src.check_human
                        .venv/bin/python -m src.check_human --fix   (rewrite the file cleanly, one JSON per line)

Checks: every row parses, has a non-empty "text", a known "theme" id (or "none"),
and only known image ids. Then reports coverage: which themes have fewer than
--min requests, so you know where to add more.
"""
import argparse
import json
from collections import Counter

from src.common import DATA, get_logger, load_tags

PATH = DATA / "requests_human.jsonl"


def parse_loose(text):
    """Read JSON objects however they are laid out (one per line, pretty-printed, comma-separated)."""
    dec, i, rows = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i] in " \n\r\t,":
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)  # raises with a position if the JSON is broken
        rows.append(obj)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=2, help="flag themes with fewer human requests than this")
    ap.add_argument("--fix", action="store_true", help="rewrite the file as clean JSONL (dedupes, [''] -> [])")
    args = ap.parse_args()
    log = get_logger("check_human")

    tags = load_tags()
    themes = {t["id"] for t in tags["themes"]} | {"none"}
    images = {i["id"] for i in tags["images"]}

    rows = parse_loose(PATH.read_text(encoding="utf-8"))
    log.info("parsed %d rows from %s", len(rows), PATH)

    errors, seen, clean = [], set(), []
    for n, r in enumerate(rows, 1):
        text = (r.get("text") or "").strip()
        imgs = [x for x in (r.get("images") or []) if x]  # drop "" left from templates
        if not text:
            errors.append(f"row {n}: empty text")
        if r.get("theme") not in themes:
            errors.append(f"row {n}: unknown theme {r.get('theme')!r}  ({text[:50]})")
        for x in imgs:
            if x not in images:
                errors.append(f"row {n}: unknown image {x!r}  ({text[:50]})")
        if text.lower() in seen:
            errors.append(f"row {n}: duplicate text  ({text[:50]})")
            continue
        seen.add(text.lower())
        clean.append({"text": text, "theme": r.get("theme"), "images": imgs})

    for e in errors:
        log.warning(e)
    log.info("errors: %d", len(errors))

    per = Counter(r["theme"] for r in clean)
    low = sorted(t for t in themes if per[t] < args.min)
    log.info("themes covered: %d/%d | rows: %d | images used: %d/%d",
             sum(1 for t in themes if per[t]), len(themes), len(clean),
             len({x for r in clean for x in r["images"]}), len(images))
    log.info("themes with < %d requests (%d): %s", args.min, len(low), ", ".join(low))

    if args.fix:
        with PATH.open("w", encoding="utf-8") as f:
            for r in clean:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("rewrote %s with %d clean rows", PATH, len(clean))


if __name__ == "__main__":
    main()
