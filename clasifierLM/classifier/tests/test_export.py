"""Check that artifacts/classifier/weights.bin holds exactly the checkpoint's
weights at the offsets meta.json claims, read the way JS reads them.
Skipped if the model hasn't been trained + exported yet."""

import json

import numpy as np
import pytest

from src.common import ARTIFACTS

META = ARTIFACTS / "classifier" / "meta.json"
pytestmark = pytest.mark.skipif(not META.exists(), reason="run `python -m src.export` first")


def test_weights_roundtrip():
    import torch

    meta = json.loads(META.read_text())
    raw = (ARTIFACTS / "classifier" / "weights.bin").read_bytes()
    sd = torch.load(ARTIFACTS / "fasttext_best.pt", weights_only=False)["state_dict"]
    names = {"embedding": "emb.weight", "theme_W": "theme_head.weight", "theme_b": "theme_head.bias",
             "image_W": "image_head.weight", "image_b": "image_head.bias",
             "hidden_W": "hidden.0.weight", "hidden_b": "hidden.0.bias"}
    for name, info in meta["arrays"].items():
        assert info["offset"] % 4 == 0, f"{name} not 4-byte aligned (JS Float32Array would throw)"
        n = int(np.prod(info["shape"]))
        arr = np.frombuffer(raw, dtype="<f4", count=n, offset=info["offset"]).reshape(info["shape"])
        np.testing.assert_array_equal(arr, sd[names[name]].numpy())


def test_meta_matches_featurizer():
    import torch

    from src.featurize import FNV_OFFSET, FNV_PRIME

    meta = json.loads(META.read_text())
    ckpt = torch.load(ARTIFACTS / "fasttext_best.pt", weights_only=False)
    assert meta["fnv"] == {"offset_basis": FNV_OFFSET, "prime": FNV_PRIME, "bits": 32}
    # B is a hyperparameter (the sweep may pick 16k/32k/65k): it must match the trained model.
    assert meta["B"] == ckpt["config"]["model"]["buckets"]
    assert meta["arrays"]["embedding"]["shape"] == [meta["B"], meta["d"]]
    assert meta["theme_ids"][-1] == "none"
