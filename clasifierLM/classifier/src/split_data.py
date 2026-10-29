"""Step 2.3: split requests_synth.jsonl into train / val / test (80/10/10).

Why three splits?
  train -- the model learns its weights from these.
  val   -- used to make choices: early stopping, the hyperparameter sweep,
           calibration thresholds. Because we *choose* based on val, val scores
           end up slightly optimistic.
  test  -- touched once at the end, to report an honest number.
  (requests_human.jsonl is a 4th, never-trained set: written by a person, so it
   shows whether a model trained on synthetic text works on real typing.)

Why "stratified by theme"? Each split gets the same theme proportions as the
whole dataset. With a plain random split, a small split (~270 rows) could by
chance get few "prem" examples and give a noisy score for that theme.

Usage:  python -m src.split_data
"""

import random
from collections import Counter

from src.common import DATA, SEED, get_logger, read_jsonl, write_jsonl

FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def main():
    log = get_logger("split_data")
    rows = read_jsonl(DATA / "requests_synth.jsonl")
    log.info("loaded %d requests", len(rows))

    splits = {name: [] for name in FRACTIONS}
    for theme in sorted({r["theme"] for r in rows}):  # sorted -> same order every run
        group = [r for r in rows if r["theme"] == theme]
        # Seeded per theme: the result doesn't depend on Python's set ordering or
        # on how many themes came before.
        random.Random(f"{SEED}-split-{theme}").shuffle(group)
        n_val = round(len(group) * FRACTIONS["val"])
        n_test = round(len(group) * FRACTIONS["test"])
        splits["val"] += group[:n_val]
        splits["test"] += group[n_val : n_val + n_test]
        splits["train"] += group[n_val + n_test :]

    for name, part in splits.items():
        # Shuffle across themes too, so files aren't grouped by theme (training
        # reshuffles anyway, but this makes the files nicer to read and inspect).
        random.Random(f"{SEED}-order-{name}").shuffle(part)
        write_jsonl(DATA / "splits" / f"{name}.jsonl", part)
        counts = Counter(r["theme"] for r in part)
        log.info("%-5s %4d rows | %s", name, len(part), dict(sorted(counts.items())))

    # Sanity check: no request text appears in two splits.
    texts = {name: {r["text"].lower() for r in part} for name, part in splits.items()}
    overlap = (texts["train"] & texts["val"]) | (texts["train"] & texts["test"]) | (texts["val"] & texts["test"])
    log.info("texts shared between splits: %d (must be 0)", len(overlap))
    assert not overlap


if __name__ == "__main__":
    main()
