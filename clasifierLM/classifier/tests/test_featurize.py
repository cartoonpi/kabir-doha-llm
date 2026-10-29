"""Tests for src/featurize.py. Run with:  pytest -v

The expected hash values are computed *independently* here (a second FNV-1a
written differently) and also checked against published reference values,
so a bug in featurize.py can't make its own test pass.
"""

from src.featurize import B, featurize, fnv1a, pieces, words


def fnv1a_reference(data: bytes) -> int:
    # Written differently on purpose: % 2**32 instead of & 0xFFFFFFFF,
    # and a while loop instead of a for loop.
    h, i = 0x811C9DC5, 0
    while i < len(data):
        h = ((h ^ data[i]) * 0x01000193) % 2**32
        i += 1
    return h


def test_fnv1a_known_values():
    # Published FNV-1a 32-bit test vectors
    assert fnv1a("") == 0x811C9DC5  # empty input = the offset basis itself
    assert fnv1a("a") == 0xE40C292C
    assert fnv1a("foobar") == 0xBF9CF968
    # And against the independent implementation
    for s in ["", "a", "<cat>", "<ca", "lamp", "ñ"]:
        assert fnv1a(s) == fnv1a_reference(s.encode("utf-8")), s


def test_fnv1a_cat_value():
    assert fnv1a("<cat>") == fnv1a_reference(b"<cat>")


def test_pieces_cat():
    assert pieces("cat") == ["<cat>", "<ca", "cat", "at>"]


def test_pieces_single_letter():
    # 1-letter words: whole-word piece only, no 3-grams
    assert pieces("a") == ["<a>"]


def test_pieces_two_letters():
    assert pieces("ok") == ["<ok>", "<ok", "ok>"]


def test_punctuation_and_case():
    assert words("Hello,  WORLD!!! (ok)\n\tbye-bye") == ["hello", "world", "ok", "bye", "bye"]
    assert pieces("!!!ok???") == pieces("ok")


def test_non_ascii_becomes_space():
    assert words("café दोहा lamp") == ["caf", "lamp"]


def test_empty_requests():
    assert featurize("") == []
    assert featurize("  ?!... ") == []


def test_buckets_in_range_and_duplicates_kept():
    ids = featurize("lamp lamp")
    assert len(ids) == 2 * len(pieces("lamp"))
    assert ids[: len(ids) // 2] == ids[len(ids) // 2 :]  # second "lamp" repeats the first
    assert all(0 <= i < B for i in ids)
