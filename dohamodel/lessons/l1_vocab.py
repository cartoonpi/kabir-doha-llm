"""Lesson 1: text becomes numbers.

A language model only sees integers. We give every distinct character an ID,
then turn the whole corpus into one long list of IDs.

Run:  .venv/bin/python lessons/l1_vocab.py
"""
import unicodedata
from collections import Counter
from pathlib import Path

text = Path(__file__).parent.parent.joinpath("data/toy_kabir.txt").read_text(encoding="utf-8")

# Unicode can spell the same Devanagari letter in more than one way (e.g. ड़ as one
# code point U+095C, or ड + a separate nukta dot ़). NFC picks one spelling (for ड़
# it is ड + ़), so the model doesn't have to learn two IDs for the same letter.
text = unicodedata.normalize("NFC", text)

dohe = [d for d in text.strip().split("\n\n")]
print(f"dohe: {len(dohe)}   characters: {len(text):,}")

# --- the vocabulary: every distinct character, sorted so IDs are stable ---
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for ch, i in stoi.items()}       # int -> string
print(f"vocabulary size: {len(chars)}")

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return "".join(itos[i] for i in ids)

# A Python "character" is a Unicode code point, not a visual letter. Vowel signs
# (ा ि ी) and the virama (्) are separate code points, so कबीर is 4, but क्या is 4 too.
for word in ["कबीर", "क्या", "प्रेम"]:
    ids = encode(word)
    print(f"{word!r:10} -> {ids}  ({len(ids)} code points: {[unicodedata.name(c).split()[-1] for c in word]})")

assert decode(encode(text)) == text  # round trip must be lossless

# --- which characters are common and which are rare? ---
counts = Counter(text)
print("\nmost common:", [(repr(c), n) for c, n in counts.most_common(8)])
rare = [c for c, n in counts.items() if n == 1]
print(f"seen only once ({len(rare)}):", " ".join(rare))
