"""Step 3: text -> list of bucket ids. Must match web/classifier.js byte-for-byte.

The idea (from fastText): instead of a fixed vocabulary of words, break each
word into small pieces and map every piece to a number with a hash function.

    "Cat!"  -> words ["cat"]
            -> pieces ["<cat>", "<ca", "cat", "at>"]
            -> hashes -> bucket ids, e.g. [9012, 431, 15877, 2203]

Why character 3-grams? They let the model handle words it never saw in
training: "lamps", "lampy" and the typo "lamp" all share the pieces "<la",
"lam", "amp", so they land in some of the same buckets.

Why "<" and ">"? They mark word edges, so "<ca" (a word *starting* with ca)
is a different piece from "ca" in the middle of "because".

Why hashing ("the hashing trick")? We never build or store a vocabulary.
Any piece, even one never seen before, maps to one of B buckets. Different
pieces sometimes collide in the same bucket, which costs a little accuracy
but keeps the model small and the browser code simple.
"""

import re

B = 16384  # number of buckets = rows in the embedding table

# 32-bit FNV-1a constants (a simple, well-known, easy-to-reimplement hash)
FNV_OFFSET = 2166136261
FNV_PRIME = 16777619

_NOT_ALLOWED = re.compile(r"[^a-z0-9 ]")


def words(text: str) -> list[str]:
    """Lowercase, turn everything except a-z, 0-9 and space into a space, split.

    Note: anything non-ASCII (accents, Devanagari, emoji) becomes a space and
    disappears. JS must do exactly the same: text.toLowerCase() then the same
    regex. (Python's str.split() with no argument also collapses whitespace.)
    """
    return _NOT_ALLOWED.sub(" ", text.lower()).split()


def pieces(text: str) -> list[str]:
    """Each word contributes "<word>" plus the 3-grams of "<word>"."""
    out = []
    for w in words(text):
        wrapped = "<" + w + ">"
        out.append(wrapped)
        if len(w) > 1:  # spec: skip 3-grams for 1-letter words ("<a>" is already 3 chars)
            out.extend(wrapped[i : i + 3] for i in range(len(wrapped) - 2))
    return out


def fnv1a(s: str) -> int:
    """32-bit FNV-1a over the UTF-8 bytes of s.

    Not Python's built-in hash(): that one is randomized per process
    (PYTHONHASHSEED), so the same word would land in a different bucket on
    every run, and JS couldn't reproduce it at all.

    `& 0xFFFFFFFF` keeps only the low 32 bits. Python ints never overflow, so
    we must truncate by hand. JS gets the same effect from Math.imul and >>> 0.
    """
    h = FNV_OFFSET
    for byte in s.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF
    return h


def featurize(text: str, n_buckets: int = B) -> list[int]:
    """The full pipeline. Duplicates are kept on purpose: a piece that appears
    twice counts twice in the model's average."""
    return [fnv1a(p) % n_buckets for p in pieces(text)]


if __name__ == "__main__":
    # python -m src.featurize "Cat, a lamp!"  -> shows each step
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Cat, a lamp!"
    print("text:  ", repr(text))
    print("words: ", words(text))
    for p in pieces(text):
        h = fnv1a(p)
        print(f"  piece {p!r:9} fnv1a={h:>10} (0x{h:08x})  bucket={h % B}")
