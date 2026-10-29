"""Step 8.1: export the trained model for the browser.

  python -m src.export          -> artifacts/classifier/       (float32)
  python -m src.export --int8   -> artifacts/classifier_int8/  (embeddings quantized)

Writes:
  weights.bin  every array back to back, little-endian, in this order:
                 embedding [B x d]   (float32, or int8 + float32 per-row scales)
                 hidden W [d x d], hidden b [d]       (only if hidden=True)
                 theme W [(T+1) x d], theme b [T+1]
                 image W [I x d], image b [I]
  meta.json    sizes, FNV constants, tag ids in head order, thresholds, and the
               byte offset/shape/dtype of every array in weights.bin.
  ../parity.json  50 test requests (+ edge cases) with Python's bucket ids and
               probabilities, for web/parity_test.mjs to compare against.

Why a raw binary file instead of JSON? 525k floats as JSON text would be ~6 MB
and slow to parse. As binary it's 2.1 MB and JS can view it in place:
  new Float32Array(buffer, byteOffset, length)  -- no parsing at all.

int8 quantization: each embedding row is stored as 8-bit integers -127..127
plus one float scale per row (row = q * scale). That cuts the embedding from 4
bytes/number to ~1, so the download shrinks ~4x, at a small accuracy cost.
"""

import argparse
import json
from datetime import datetime

import numpy as np
import torch

from src.calibrate import decide
from src.common import ARTIFACTS, DATA, get_logger, read_jsonl
from src.featurize import FNV_OFFSET, FNV_PRIME, featurize
from src.model import FastTextPredictor

# Edge cases worth checking on both sides, besides real test requests.
EDGE_CASES = [
    "", "?!?...", "a", "Cat, a LAMP!!", "lamp lamp lamp", "tab\tand\nnewline",
    "Café NAÏVE — lamp", "दोहा about death", "emoji 🙂 fish 🐟", "İstanbul market", "x" * 300,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true", help="per-row int8 embeddings")
    args = ap.parse_args()
    log = get_logger("export_int8" if args.int8 else "export")

    pred = FastTextPredictor(ARTIFACTS / "fasttext_best.pt")
    sd = {k: v.numpy() for k, v in pred.model.state_dict().items()}
    cfg = pred.cfg["model"]
    thresholds = json.loads((ARTIFACTS / "thresholds.json").read_text())["fasttext"]
    out_dir = ARTIFACTS / ("classifier_int8" if args.int8 else "classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = []  # (name, numpy array) in file order
    emb = sd["emb.weight"].astype(np.float32)
    if args.int8:
        scale = np.abs(emb).max(axis=1) / 127.0
        scale[scale == 0] = 1.0  # unused all-zero rows: avoid dividing by 0
        q = np.clip(np.round(emb / scale[:, None]), -127, 127).astype(np.int8)
        arrays += [("embedding_q", q), ("embedding_scale", scale.astype(np.float32))]
        log.info("int8: max abs reconstruction error %.2e", np.abs(q * scale[:, None] - emb).max())
    else:
        arrays.append(("embedding", emb))
    if cfg["hidden"]:
        arrays += [("hidden_W", sd["hidden.0.weight"]), ("hidden_b", sd["hidden.0.bias"])]
    arrays += [("theme_W", sd["theme_head.weight"]), ("theme_b", sd["theme_head.bias"]),
               ("image_W", sd["image_head.weight"]), ("image_b", sd["image_head.bias"])]

    layout, chunks, offset = {}, [], 0
    for name, arr in arrays:
        arr = np.ascontiguousarray(arr, dtype=arr.dtype.newbyteorder("<"))  # force little-endian
        layout[name] = {"offset": offset, "shape": list(arr.shape), "dtype": str(arr.dtype.name)}
        data = arr.tobytes()
        # Pad to a multiple of 4 bytes: a JS Float32Array must start at an
        # offset divisible by 4, and an int8 block can end anywhere.
        data += b"\0" * (-len(data) % 4)
        chunks.append(data)
        offset += len(data)
    (out_dir / "weights.bin").write_bytes(b"".join(chunks))

    meta = {
        "format": "doha-tag-classifier/1",
        "B": cfg["buckets"], "d": cfg["dim"], "hidden": cfg["hidden"], "int8": args.int8,
        "fnv": {"offset_basis": FNV_OFFSET, "prime": FNV_PRIME, "bits": 32},
        "featurize": "lowercase; [^a-z0-9 ] -> space; split; '<w>' + char 3-grams of '<w>' (none if len(w)==1); fnv1a(utf8) % B",
        "theme_ids": pred.theme_ids, "image_ids": pred.image_ids,
        "thresholds": {k: thresholds[k] for k in ("t_theme", "t_img", "max_images")},
        "arrays": layout,
        "exported": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("wrote %s: weights.bin %.2f MB, meta.json", out_dir.relative_to(ARTIFACTS.parent), offset / 1e6)
    for name, info in layout.items():
        log.info("  %-16s offset %9d  shape %-12s %s", name, info["offset"], info["shape"], info["dtype"])

    # Parity file: the Python side of the Python-vs-JS check. Probabilities come
    # from the real PyTorch model (float32), the reference JS must match.
    if not args.int8:
        texts = [r["text"] for r in read_jsonl(DATA / "splits/test.jsonl")[:50]] + EDGE_CASES
        cases = []
        for text, p in zip(texts, pred.predict(texts)):
            cases.append({"text": text, "buckets": featurize(text, cfg["buckets"]),
                          "theme_probs": p["theme_probs"], "image_probs": p["image_probs"],
                          "decision": decide(p, thresholds["t_theme"], thresholds["t_img"])})
        (ARTIFACTS / "parity.json").write_text(json.dumps({"cases": cases}, ensure_ascii=False) + "\n")
        log.info("wrote artifacts/parity.json: %d cases (50 test + %d edge cases)", len(cases), len(EDGE_CASES))


if __name__ == "__main__":
    main()
