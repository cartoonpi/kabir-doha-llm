"""Step 6: choose decision thresholds on the VAL set.

A model outputs probabilities; an app needs decisions. Two thresholds turn one
into the other:

  t_theme -- "no clear match": if the top theme's probability is below t_theme,
             answer "none" instead. Softmax always picks *something*, even for
             "what's the weather?". This lets low confidence mean "none".
  t_img   -- keep an image if its sigmoid probability >= t_img (at most 2,
             because requests have 0-2 images).

Why val and not test? Choosing a threshold is a form of training: we pick the
value that scores best. Doing that on test would make the test score optimistic.
(Val was also used for early stopping, so val scores are already a bit rosy;
test stays untouched until step 7.)

Usage:  python -m src.calibrate   -> artifacts/thresholds.json
"""

import json

import numpy as np
from sklearn.metrics import f1_score

from src.common import ARTIFACTS, DATA, get_logger, load_tags, read_jsonl

MAX_IMAGES = 2
T_THEME_GRID = np.round(np.arange(0.20, 0.801, 0.05), 2)
T_IMG_GRID = np.round(np.arange(0.30, 0.801, 0.05), 2)


def decide(pred: dict, t_theme: float, t_img: float) -> dict:
    """Probabilities -> final tags. The ONE place this rule lives (predict.py,
    evaluate.py and, mirrored, web/classifier.js all use it)."""
    tp = pred["theme_probs"]
    theme = max(tp, key=tp.get)
    if tp[theme] < t_theme:
        theme = "none"
    ranked = sorted(pred["image_probs"].items(), key=lambda kv: -kv[1])
    top = [i for i, p in ranked if p >= t_img][:MAX_IMAGES]  # the (up to 2) most likely images
    # Return in tag-file order, not probability order. Two images at ~1.0 can
    # tie in float32 (PyTorch) but not in float64 (JS), which would make the
    # order differ between Python and the browser (caught by the parity test).
    order = list(pred["image_probs"])
    return {"theme": theme, "images": sorted(top, key=order.index)}


def theme_macro_f1(rows, preds, t):
    true = [r["theme"] for r in rows]
    pred = [decide(p, t, 1.0)["theme"] for p in preds]
    return f1_score(true, pred, average="macro", zero_division=0)  # "none" counts as a class


def image_micro_f1(rows, preds, t, image_ids):
    Y = np.array([[i in r["images"] for i in image_ids] for r in rows])
    P = np.array([[i in decide(p, 1.0, t)["images"] for i in image_ids] for p in preds])
    return f1_score(Y, P, average="micro", zero_division=0)


def calibrate(name, preds, rows, image_ids, log):
    th_scores = [theme_macro_f1(rows, preds, t) for t in T_THEME_GRID]
    im_scores = [image_micro_f1(rows, preds, t, image_ids) for t in T_IMG_GRID]
    # np.argmax returns the FIRST best -> on ties, the lowest threshold wins.
    t_theme = float(T_THEME_GRID[int(np.argmax(th_scores))])
    t_img = float(T_IMG_GRID[int(np.argmax(im_scores))])

    log.info("[%s] t_theme sweep (val theme macro-F1, 'none' counted as a class):", name)
    for t, s in zip(T_THEME_GRID, th_scores):
        log.info("    t_theme=%.2f  macro-F1 %.4f %s", t, s, "<- best" if t == t_theme else "")
    log.info("[%s] t_img sweep (val image micro-F1, max %d images):", name, MAX_IMAGES)
    for t, s in zip(T_IMG_GRID, im_scores):
        log.info("    t_img=%.2f    micro-F1 %.4f %s", t, s, "<- best" if t == t_img else "")

    # For context: how much did calibration change things vs the naive rule
    # (argmax theme with no threshold, images at 0.5)?
    naive_t = theme_macro_f1(rows, preds, 0.0)
    naive_i = image_micro_f1(rows, preds, 0.5, image_ids)
    log.info("[%s] chosen t_theme=%.2f t_img=%.2f | theme macro-F1 %.4f -> %.4f | image micro-F1 %.4f -> %.4f",
             name, t_theme, t_img, naive_t, max(th_scores), naive_i, max(im_scores))
    return {"t_theme": t_theme, "t_img": t_img, "max_images": MAX_IMAGES,
            "val_theme_macroF1": round(max(th_scores), 4), "val_image_microF1": round(max(im_scores), 4)}


def main():
    from src.baselines import LogRegModel
    from src.model import FastTextPredictor

    log = get_logger("calibrate")
    tags = load_tags()
    image_ids = [i["id"] for i in tags["images"]]
    train, val = read_jsonl(DATA / "splits/train.jsonl"), read_jsonl(DATA / "splits/val.jsonl")
    texts = [r["text"] for r in val]

    models = {
        "fasttext": FastTextPredictor(ARTIFACTS / "fasttext_best.pt"),
        "logreg": LogRegModel(tags).fit(train),  # deterministic, retrains in ~1 s
    }
    out = {name: calibrate(name, m.predict(texts), val, image_ids, log) for name, m in models.items()}

    path = ARTIFACTS / "thresholds.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    log.info("wrote %s:\n%s", path.relative_to(ARTIFACTS.parent), json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
