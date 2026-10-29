"""Export trained snapshots for the browser page (web/).

For each checkpoints/<name>/step_<N>.pt  ->  web/models/<name>/step_<N>.bin
  float32 little-endian, every trained tensor back to back (the causal mask buffer is skipped;
  JS rebuilds it). web/models/<name>/meta.json says where each tensor starts.

meta.json also carries the vocabulary, the training curve, the train/val dohe (for the copy
check and example prompts) and, per snapshot, PyTorch's next-char logits for a few prompts,
so web/parity_test.mjs can check the JS forward pass gives the same numbers.

Run:  .venv/bin/python export.py small big
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from model import GPT, Config  # noqa: E402

PARITY_PROMPTS = ["बुरा जो देखन", "\n", "माया मरी न मन मरा, मर मर गए शरीर।\n"]


def export(name):
    src = ROOT / "checkpoints" / name
    info = json.loads((src / "log.json").read_text(encoding="utf-8"))
    cfg = Config(**info["config"])
    stoi = {c: i for i, c in enumerate(info["itos"])}
    out = ROOT / "web" / "models" / name
    out.mkdir(parents=True, exist_ok=True)

    tensors, parity = None, {}
    for step in info["snapshots"]:
        sd = torch.load(src / f"step_{step}.pt")
        keep = {k: v for k, v in sd.items() if not k.endswith(".mask")}
        offsets, chunks, pos = {}, [], 0
        for k, v in keep.items():
            a = v.detach().float().numpy().astype("<f4").ravel()
            offsets[k] = {"offset": pos, "shape": list(v.shape)}   # offset counted in floats
            chunks.append(a)
            pos += a.size
        (out / f"step_{step}.bin").write_bytes(np.concatenate(chunks).tobytes())
        tensors = tensors or offsets  # same layout for every snapshot

        model = GPT(cfg)
        model.load_state_dict(sd)
        model.eval()
        parity[step] = []
        with torch.no_grad():
            for p in PARITY_PROMPTS:
                ids = torch.tensor([[stoi[c] for c in p]])
                logits, _ = model(ids)
                parity[step].append({"prompt": p, "logits": logits[0, -1].tolist()})
        print(f"{name} step {step}: {pos:,} floats ({pos * 4 / 1e6:.1f} MB)")

    meta = {k: info[k] for k in ("config", "log", "best", "snapshots", "itos", "train_dohe", "val_dohe")}
    meta.update(name=name, n_params=GPT(cfg).n_params(), tensors=tensors, parity=parity)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        export(n)
