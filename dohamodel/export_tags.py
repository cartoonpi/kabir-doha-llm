"""Export Lesson 8 tag-steered models for the browser (web/steer.html).

checkpoints/<name>/stage2.pt  ->  web/models/<name>/step_final.bin   (same float32 layout as export.py,
so web/gpt.js loads it unchanged: loadModel(base, "final", meta))

meta.json carries what the page needs besides the weights:
  itos (characters, then "|", then the tag tokens), first_tag (ids >= this are never shown),
  the tag lists with English names (themes, images, devices), the training curve and metrics,
  the real dohe (for the page's copy check), and PyTorch's next-token logits for a few tagged
  prompts, so web/steer_parity.mjs can check the JS forward pass.

Run:  .venv/bin/python export_tags.py tags_5k tags_2500
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
from model import GPT, Config  # noqa: E402
from synth_pilot import DEVICES, IMAGES, THEMES  # noqa: E402
from tag_real import IMAGE_WORDS  # noqa: E402

PARITY = [["<mrityu>", "<deepak>", "<kahekabir>", "|"], ["<guru>", "<kumhar>", "<sadho>", "|", "ग", "ु", "र"], ["|"]]


def export(name):
    src = ROOT / "checkpoints" / name
    info = json.loads((src / "log.json").read_text(encoding="utf-8"))
    cfg = Config(**info["config"])
    itos = info["itos"]
    stoi = {s: i for i, s in enumerate(itos)}
    out = ROOT / "web" / "models" / name
    out.mkdir(parents=True, exist_ok=True)

    sd = torch.load(src / "stage2.pt", map_location="cpu")
    keep = {k: v for k, v in sd.items() if not k.endswith(".mask")}
    offsets, chunks, pos = {}, [], 0
    for k, v in keep.items():
        a = v.detach().float().numpy().astype("<f4").ravel()
        offsets[k] = {"offset": pos, "shape": list(v.shape)}
        chunks.append(a)
        pos += a.size
    (out / "step_final.bin").write_bytes(np.concatenate(chunks).tobytes())

    model = GPT(cfg)
    model.load_state_dict(sd)
    model.eval()
    parity = []
    with torch.no_grad():
        for toks in PARITY:
            ids = [stoi["\n"], stoi["\n"]] + [stoi[t] for t in toks]
            logits, _ = model(torch.tensor([ids]))
            parity.append({"ids": ids, "logits": logits[0, -1].tolist()})

    real = [json.loads(l)["text"] for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")
            if json.loads(l)["split"] != "drop"]
    meta = {
        "name": name, "config": info["config"], "args": info["args"], "log": info["log"], "best": info["best"],
        "metrics": info["metrics"], "itos": itos, "first_tag": stoi["|"], "snapshots": ["final"],
        "themes": [{"id": t["id"], "name": t["name"], "desc": t["desc"]} for t in THEMES],
        "images": [{"id": i["id"], "name": i["name"], "words": IMAGE_WORDS[i["id"]].split()} for i in IMAGES],
        "devices": [{"id": d["id"], "name": d["name"].split(":")[0]} for d in DEVICES],
        "real_dohe": real, "n_params": GPT(cfg).n_params(), "tensors": offsets, "parity": parity,
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"{name}: {pos:,} floats ({pos * 4 / 1e6:.1f} MB), {len(itos)} tokens, val {info['best']['stage2']['val']:.3f}")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        export(n)
