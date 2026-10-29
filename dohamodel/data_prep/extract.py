"""Step 1 of real data: pull dohe out of the Belvedere Press OCR text.

Input : data/raw/belvedere_<year>.txt   (archive.org OCR, one printed line per line)
Output: data/interim/belvedere_<year>.jsonl, one doha per line:
    {"i": running index, "section": "चितावनी", "num": 48 or null, "raw": [line1, line2],
     "text": "clean line 1।\nclean line 2॥", "src_lines": [n1, n2]}

How a printed page looks after OCR (see lesson notes):
    आये हैं. से जायेंगे, राजा रंकः फकीर।           <- doha line 1: has a comma, ends in ।
    0) अगर । (२) घरदरा । (३) घड़ा मिद्दी का।          <- footnotes: "(N)" glosses
    छ४ कवीर सांखी-लंग्रह                               <- running header (book title)
    एक सिंघासन चढ़ि चले, इक बाँधे जात जेंजीर ॥४८॥    <- doha line 2: has a comma, ends in ॥N॥
So we keep only "verse-looking" lines, and pair a line without ॥ with the next line that has one.

Run:  .venv/bin/python data_prep/extract.py 1926 1912
"""
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent

# The book's sections in printed order (read off the running headers of the 1926 OCR and
# the table of contents, OCR errors fixed by hand). Headers are matched to this list fuzzily.
SECTIONS = """गुरुदेव|झूठे गुरु|गुरुमुख|निगुरा|गुरु शिष्य खोज|सेवक और दास|सूरमा|पतिव्रता|सती|बिभिचारिन|भक्ति
|लौ|बिरह|प्रेम|सतसंग|कुसंग|सूक्ष्म मार्ग|चितावनी|सहज|बिश्वास|मध्य|बाचक ज्ञान|अनुभव ज्ञान|सार गहनी|अपारख
|नाम|सुमिरन|शब्द|उपदेश|सामर्थ|निज करता का निर्णय|घट मठ|समदृष्टि|भेदी|परिचय|मौन|सजीवन|जीवत मृतक|साध
|बेहद|भेष|असाधु|बैरागी की रहनी|काम|क्रोध|लोभ|मोह|मान और हंगता|कपट|आशा|तृष्णा|शील|क्षमा|धीरज|दीनता|दया
|बिचार|बिबेक|बुद्धि और कुबुद्धि|मन|माया|कनक और कामिनी|निद्रा|निंदा|माँस अहार|स्वादिष्ट भोजन|नशा
|पर त्रिया|तीर्थ ब्रत|पंडित और संस्कृत|मिश्रित""".replace("\n", "").split("|")

DEV = re.compile(r"[ऀ-ॿ]")
# characters a clean doha may contain: Devanagari letters and signs, space, comma
KEEP = re.compile(r"[^ऀ-ॐक़-ॣॱ-ॿ ,]")   # also drops Vedic accents U+0951-0957
DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm(s):
    return unicodedata.normalize("NFC", s).replace("|", "।").strip()


def dev_ratio(s):
    letters = [c for c in s if not c.isspace()]
    return sum(bool(DEV.match(c)) for c in letters) / max(1, len(letters))


def is_footnote(s):
    # "(१) पालन पोषन।"  or  "0) अगर ।"  (OCR often drops the opening bracket)
    return bool(re.search(r"\(\s*[०-९\d]{1,2}\s*\)", s) or re.match(r"^\W{0,2}[०-९\d]{1,2}\s*\)", s))


def header_name(s):
    """'खितावनी का अंग है३' -> 'खितावनी';  None if the line is not a section header."""
    if "," in s or len(s) > 60:
        return None
    m = re.search(r"(.*?)\s*क[ा!ं]?\s*[-]?\s*अंग", s)
    if not m:
        return None
    name = re.sub(r"[^ऀ-ॣॱ-ॿ ]", " ", m.group(1))     # drop digits, punctuation
    name = re.sub(r"^[\s०-९]*", "", name).strip()
    return name or None


CAESURA_SUB = re.compile(r"(?<=\S)[?!)](?=\s)")   # OCR often reads the mid-line comma as ? ! )
NUM_ONLY = re.compile(r"^[\s॥।'\"*.\-]*[०-९\d][^\s]{0,5}[\s॥।'\"*.\-]*$|^\s*॥[^\s]{1,6}\s*$")


def fix_caesura(s):
    return s if "," in s else CAESURA_SUB.sub(",", s, count=1)


def verse_like(s):
    """A printed doha line: mostly Devanagari, one-line length, with a mid-line comma
    (or a line-ending mark, for the rare line whose comma OCR lost entirely)."""
    if not (12 <= len(s) <= 100 and dev_ratio(s) > 0.85):
        return False
    return "," in s[3:-3] or line_type(s) != "?"


def line_type(s):
    """'1' = first line (ends in ।), '2' = second line (ends in the verse number), '?' = can't tell."""
    t = s.rstrip(" '\"*!.-_:;")
    tail = t.split()[-1] if t.split() else ""
    if "॥" in tail or re.search(r"[०-९\d]", tail):
        return "2"
    if t.endswith("।"):
        return "1"
    return "?"


def split_number(s):
    """'... जेंजीर ॥४८॥' -> ('... जेंजीर', 48)."""
    m = re.search(r"\s*[।॥].*$", s)
    body, tail = (s[: m.start()], s[m.start():]) if m else (s, "")
    digits = re.sub(r"\D", "", tail.translate(DIGITS))
    clean_tail = re.fullmatch(r"\s*॥\s*[०-९]+\s*॥\s*", tail) is not None
    return body, (int(digits) if digits and clean_tail else None)


# Old printing fonts built ो/ौ from two pieces (े + ा); OCR keeps the pieces apart.
PIECES = {"\u0947\u093e": "\u094b", "\u093e\u0947": "\u094b", "\u0948\u093e": "\u094c", "\u093e\u0948": "\u094c"}


def clean(s):
    s = KEEP.sub("", s)
    for a, b in PIECES.items():
        s = s.replace(a, b)
    s = re.sub(r"\s*,\s*", ", ", s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def extract(year):
    lines = [norm(l) for l in (ROOT / f"data/raw/belvedere_{year}.txt").read_text(encoding="utf-8").splitlines()]

    # skip front matter: start at the first header that matches the first section
    start = next(i for i, l in enumerate(lines)
                 if (h := header_name(l)) and difflib.SequenceMatcher(None, h, SECTIONS[0]).ratio() > 0.7)

    # pass 1: walk the lines, track the section, keep verse candidates
    sec_idx, cands, stats = 0, [], Counter()
    for n in range(start, len(lines)):
        s = lines[n]
        if not s:
            continue
        if (h := header_name(s)) is not None:
            # match only the current section or the next few (books go forward)
            window = SECTIONS[sec_idx: sec_idx + 4]
            best = max(range(len(window)), key=lambda k: difflib.SequenceMatcher(None, h, window[k]).ratio())
            if difflib.SequenceMatcher(None, h, window[best]).ratio() >= 0.5:
                sec_idx += best
            stats["header"] += 1
        elif is_footnote(s):
            stats["footnote"] += 1
        elif NUM_ONLY.match(s) and cands:
            # a verse number printed on its own line belongs to the line before it
            cands[-1]["type"] = "2"
            cands[-1]["s"] += " " + s
            stats["number_only"] += 1
        elif verse_like(s := fix_caesura(s)):
            cands.append({"n": n, "s": s, "type": line_type(s), "section": SECTIONS[sec_idx]})
        else:
            stats["other"] += 1

    # pass 2: pair line 1 with line 2. A '1' or '?' followed (within a page or so) by a '2' or '?'.
    out, i = [], 0
    while i < len(cands):
        a = cands[i]
        b = cands[i + 1] if i + 1 < len(cands) else None
        too_short = b and min(len(a["s"]), len(b["s"])) < 18   # half-lines from broken page layout
        if a["type"] != "2" and b and b["type"] != "1" and b["n"] - a["n"] <= 15 and not too_short:
            body2, num = split_number(b["s"])
            body1 = re.sub(r"\s*[।|]\W*$", "", a["s"])
            out.append({"i": len(out), "section": a["section"], "num": num, "raw": [a["s"], b["s"]],
                        "text": f"{clean(body1)}।\n{clean(body2)}॥", "src_lines": [a["n"] + 1, b["n"] + 1]})
            i += 2
        else:
            stats["orphan_line" + ("2" if a["type"] == "2" else "1")] += 1
            i += 1

    # self-check: inside a section, readable verse numbers should go up by exactly 1
    steps = Counter()
    for x, y in zip(out, out[1:]):
        if x["num"] and y["num"] and x["section"] == y["section"]:
            steps["+1" if y["num"] == x["num"] + 1 else "other"] += 1
    stats["numbers_+1"], stats["numbers_jump"] = steps["+1"], steps["other"]

    dst = ROOT / f"data/interim/belvedere_{year}.jsonl"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in out), encoding="utf-8")

    per_sec = Counter(d["section"] for d in out)
    print(f"\n== {year}: {len(out)} dohe  (front matter skipped: {start} lines)")
    print("   line kinds:", dict(stats))
    print(f"   verse number readable: {sum(d['num'] is not None for d in out)}/{len(out)}")
    print(f"   sections found: {len(per_sec)}/{len(SECTIONS)}   missing: {[s for s in SECTIONS if s not in per_sec]}")
    return out


if __name__ == "__main__":
    for y in sys.argv[1:] or ["1926", "1912"]:
        extract(y)
