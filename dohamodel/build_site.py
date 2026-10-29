"""Build the production site (dohellm/site/) for Cloudflare Pages.

Sources (read-only):
  web/models/tags_5k/            the doha model (Lesson 8, 1.8M params)
  web/gpt.js, web/matra.js       forward pass and meter checker
  ../clasifierLM/classifier/     the other agent's classifier (web/classifier.js, artifacts/classifier/)

Output: site/ with weights stored as float16 (half the download; the page widens them back to
float32 on load), trimmed meta files, and a cache-busting version for the service worker.
Hand-written files at site/ root (index.html, app.js, worker.js, sw.js, ...) are left alone; this script only fills
site/lib/ and site/models/ and stamps the version.

Run:  .venv/bin/python build_site.py        then check (from dohellm/):  node dohamodel/site_check.mjs
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
SITE = ROOT.parent / "site"
CLF = ROOT.parent / "clasifierLM" / "classifier"
MODEL = ROOT / "web" / "models" / "tags_5k"


def f16(path, out):
    a = np.frombuffer(path.read_bytes(), dtype="<f4")
    out.write_bytes(a.astype("<f2").tobytes())
    return a.size


def main():
    (SITE / "lib").mkdir(parents=True, exist_ok=True)
    (SITE / "models").mkdir(exist_ok=True)

    # libraries: gpt.js gets one change (export the class so the worker can build it from a buffer)
    gpt = (ROOT / "web" / "gpt.js").read_text(encoding="utf-8").replace("\nclass GPT {", "\nexport class GPT {")
    (SITE / "lib" / "gpt.js").write_text(gpt, encoding="utf-8")
    shutil.copy(ROOT / "web" / "matra.js", SITE / "lib" / "matra.js")
    shutil.copy(CLF / "web" / "classifier.js", SITE / "lib" / "classifier.js")

    # doha model
    m = json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))
    n = f16(MODEL / "step_final.bin", SITE / "models" / "doha.bin")
    keep = {k: m[k] for k in ("config", "itos", "first_tag", "themes", "images", "devices", "tensors", "real_dohe", "parity")}
    keep["floats"] = n
    (SITE / "models" / "doha.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")

    # classifier: byte offsets in its meta stay valid once the page widens float16 back to float32
    cm = json.loads((CLF / "artifacts" / "classifier" / "meta.json").read_text(encoding="utf-8"))
    assert not cm["int8"] and all(a["dtype"] == "float32" for a in cm["arrays"].values())
    cm["floats"] = f16(CLF / "artifacts" / "classifier" / "weights.bin", SITE / "models" / "classifier.bin")
    (SITE / "models" / "classifier.json").write_text(json.dumps(cm), encoding="utf-8")

    # version = hash of everything the service worker caches
    h = hashlib.sha256()
    for p in sorted(SITE.rglob("*")):
        if p.is_file() and p.name not in ("sw.js",):
            h.update(p.read_bytes())
    ver = h.hexdigest()[:10]
    sw = SITE / "sw.js"
    sw.write_text(re.sub(r'const VERSION = "[^"]*";', f'const VERSION = "{ver}";', sw.read_text(encoding="utf-8")), encoding="utf-8")
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"site/ built: version {ver}, {size / 1e6:.1f} MB total")


if __name__ == "__main__":
    main()
