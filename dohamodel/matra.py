"""Count matras (rhythmic beats) in Devanagari verse, and check the doha meter.

A doha has 2 lines; each line is two halves (charan) split at the caesura (the comma):
13 matras + 11 matras.  This version implements exactly the PRD's simplified rules:

  1. A syllable with a short vowel = 1 (laghu): अ इ उ ऋ, signs ि ु ृ, or no sign (inherent अ).
  2. A syllable with a long vowel = 2 (guru): आ ई ऊ ए ऐ ओ औ, signs ा ी ू े ै ो ौ.
  3. A short syllable becomes 2 if the NEXT consonant in the same word is a joined one
     (consonant + virama ् + consonant, e.g. सत्य: स = 2), or if it carries anusvara ं or visarga ः.
  4. Chandrabindu ँ adds nothing.

Tuning options (Lesson 6), both off by default:
  nasal=True  word-final ं after a short vowel sign (ि ु ृ) is only nasalization (like ँ), so it adds
              nothing: मनहिं = 1+1+1. Before a consonant it is a real nasal: निंदक = 2+1+1.
  flex=True   metrical licence: some long syllables may be read short, so a half has a RANGE of
              counts, and it passes if the target is inside the range. See FLEX_WORDS.

Every written syllable counts, including a word-final consonant (कबीर = क 1 + बी 2 + र 1 = 4),
as in traditional chhand counting. The rules have known edge cases; lessons/l6_matra.py measures
how often they agree with real dohe, and we tune from there.
"""
import re

CONS = set(chr(c) for c in range(0x0915, 0x093A)) | set("क़ख़ग़ज़ड़ढ़फ़य़")
NUKTA, VIRAMA, ANUSVARA, VISARGA, CHANDRA = "़", "्", "ं", "ः", "ँ"
SHORT_SIGNS, LONG_SIGNS = set("िुृ"), set("ाीूेैोौ")
SHORT_VOWELS, LONG_VOWELS = set("अइउऋ"), set("आईऊएऐओऔ")


# short words whose long vowel poets often read short, and any word-final ै (कहै, जरै, रहै)
FLEX_WORDS = {"है", "हैं", "मैं", "में", "के", "से", "जो", "सो", "ते", "को", "तो", "ने", "जे", "कै", "सों", "तें"}


def syllables(word, nasal=False):
    """Split a word into syllables with their weight: [('दी', 2), ('प', 1), ('क', 1)].

    A syllable = (consonant cluster) + vowel. A consonant followed by virama joins the
    next consonant (the cluster), and makes the PREVIOUS syllable long (rule 3)."""
    out, i, n = [], 0, len(word)
    while i < n:
        start, c = i, word[i]
        if c in CONS:
            i += 1
            if i < n and word[i] == NUKTA:
                i += 1
            # cluster: consonant ् consonant ...  (the first consonant has no vowel of its own)
            while i + 1 < n and word[i] == VIRAMA and word[i + 1] in CONS:
                if out and start > 0 and not out[-1][2]:
                    out[-1][1] = 2                        # rule 3: joined consonant lengthens the previous syllable
                    out[-1][2] = True
                i += 2
                if i < n and word[i] == NUKTA:
                    i += 1
            if i < n and word[i] == VIRAMA:                # word-final half consonant (rare): no syllable
                i += 1
                continue
            if i < n and word[i] in LONG_SIGNS:
                w, i = 2, i + 1
            elif i < n and word[i] in SHORT_SIGNS:
                w, i = 1, i + 1
            else:
                w = 1                                      # inherent अ
        elif c in LONG_VOWELS:
            w, i = 2, i + 1
        elif c in SHORT_VOWELS:
            w, i = 1, i + 1
        else:
            i += 1                                         # stray sign or punctuation: ignore
            continue
        locked = w == 2
        after_short_sign = word[i - 1] in SHORT_SIGNS if i > start else False
        while i < n and word[i] in (ANUSVARA, VISARGA, CHANDRA):
            if word[i] == ANUSVARA and nasal and after_short_sign and i == n - 1:
                pass                                       # nasal rule: word-final हिं counts like हिँ
            elif word[i] in (ANUSVARA, VISARGA):
                w, locked = 2, True                        # rule 3: anusvara / visarga
            i += 1                                         # rule 4: chandrabindu adds nothing
        out.append([word[start:i], w, locked])
    return [(s, w) for s, w, _ in out]


WORDS = re.compile(r"[ऀ-ॣॱ-ॿ]+")


def count(text, nasal=False):
    """Total matras in a piece of text (words separated by spaces/punctuation)."""
    return sum(w for word in WORDS.findall(text) for _, w in syllables(word, nasal))


def count_range(text, nasal=False, flex=False):
    """(min, max) matras, letting flexible long syllables count 1 or 2."""
    lo = hi = 0
    for word in WORDS.findall(text):
        syl = syllables(word, nasal)
        for k, (s, w) in enumerate(syl):
            hi += w
            bendable = flex and w == 2 and (word in FLEX_WORDS or (k == len(syl) - 1 and s.endswith("ै") and len(syl) > 1))
            lo += 1 if bendable else w
    return lo, hi


def offsets(text, nasal=False, flex=False):
    """How far each half is from its target, [l1a, l1b, l2a, l2b]: 0 if the target is inside the half's
    range, else the distance (+2 = two matras too long). None if the doha has no 2-line, comma shape."""
    lines = text.strip().split("\n")
    if len(lines) != 2:
        return None
    out = []
    for line in lines:
        line = line.strip(" ।॥")
        if "," not in line:
            return None
        for half, target in zip(line.split(",", 1), (13, 11)):
            lo, hi = count_range(half, nasal, flex)
            out.append(0 if lo <= target <= hi else (lo - target if lo > target else hi - target))
    return out


def check(text, nasal=False, flex=False, slack=0):
    """True if both lines scan as 13 + 11 (requires the caesura comma).
    slack=1: one half (of the four) may be off by 1 matra. Lesson 6c: accepts 86% of real Kabir
    (61% with slack=0) and 8% of broken dohe (3%); the counter itself misreads some real lines."""
    o = offsets(text, nasal, flex)
    if o is None:
        return False
    bad = [x for x in o if x]
    return not bad or (slack >= 1 and len(bad) == 1 and abs(bad[0]) == 1)


def grade(text):
    """'exact' (all four halves scan), 'near' (one half off by 1) or None, with nasal + flex on."""
    if check(text, nasal=True, flex=True):
        return "exact"
    return "near" if check(text, nasal=True, flex=True, slack=1) else None


def scan(text):
    """Per line, per half: matra counts. 'a, b।\\nc, d॥' -> [(13, 11), (13, 11)].
    A line with no comma gives (total, None)."""
    out = []
    for line in text.strip().split("\n"):
        line = line.strip(" ।॥")
        if "," in line:
            a, b = line.split(",", 1)
            out.append((count(a), count(b)))
        else:
            out.append((count(line), None))
    return out


def is_doha(text):
    s = scan(text)
    return len(s) == 2 and all(h == (13, 11) for h in s)


if __name__ == "__main__":
    for w in ["कबीर", "दीपक", "सत्य", "प्रेम", "संत", "गाँव", "दुःख"]:
        print(w, syllables(w), count(w))
    print(scan("दीपक जलता देह का, ..."))
