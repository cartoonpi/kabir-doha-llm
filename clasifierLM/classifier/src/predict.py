"""Step 9: command-line prediction.

  python -m src.predict "something about how life slowly runs out, use a lamp"
  python -m src.predict --model logreg --top 3 "my boss thinks he knows everything"

Output:  theme: mrityu (0.57)   images: deepak (0.98)
"""

import argparse
import json

from src.baselines import KeywordModel, LogRegModel
from src.calibrate import decide
from src.common import ARTIFACTS, DATA, load_tags, read_jsonl


def main():
    ap = argparse.ArgumentParser(description="English request -> doha tags")
    ap.add_argument("text")
    ap.add_argument("--model", choices=["keyword", "logreg", "fasttext"], default="fasttext")
    ap.add_argument("--top", type=int, default=0, help="also show the top N themes with probabilities")
    args = ap.parse_args()

    tags = load_tags()
    thresholds = json.loads((ARTIFACTS / "thresholds.json").read_text())
    if args.model == "fasttext":
        from src.model import FastTextPredictor  # imported here: torch is slow to import

        model = FastTextPredictor(ARTIFACTS / "fasttext_best.pt")
    elif args.model == "logreg":
        model = LogRegModel(tags).fit(read_jsonl(DATA / "splits/train.jsonl"))  # ~1 s, deterministic
    else:
        model = KeywordModel(tags)

    p = model.predict([args.text])[0]
    if args.model == "keyword":
        tp = p["theme_probs"]
        d = {"theme": max(tp, key=tp.get), "images": [i for i, v in p["image_probs"].items() if v > 0]}
    else:
        t = thresholds[args.model]
        d = decide(p, t["t_theme"], t["t_img"])

    images = ", ".join(f"{i} ({p['image_probs'][i]:.2f})" for i in d["images"]) or "-"
    print(f"theme: {d['theme']} ({p['theme_probs'][d['theme']]:.2f})   images: {images}")
    if args.top:
        for t, prob in sorted(p["theme_probs"].items(), key=lambda kv: -kv[1])[: args.top]:
            print(f"  {t:8} {prob:.3f} {'#' * round(prob * 40)}")


if __name__ == "__main__":
    main()
