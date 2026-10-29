"""Step 5.2: train the fastText-style classifier.

  python -m src.train            # one run with config/train.yaml
  python -m src.train --adam     # fallback optimizer: Adam, lr 1e-2
  python -m src.train --sweep    # try every combination in train.yaml's sweep

The best checkpoint (by val theme macro-F1) goes to artifacts/fasttext_best.pt.

One training step, in plain words:
  1. featurize a batch of 32 requests, run them through the model -> scores
  2. loss = how wrong the scores are (theme cross-entropy + λ * image BCE)
  3. loss.backward() computes, for every weight, which direction lowers the loss
  4. optimizer.step() moves every weight a little (lr) in that direction
"""

import argparse
import copy
import itertools
import random
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.common import ARTIFACTS, CONFIG, DATA, REPORTS, SEED, get_logger, load_tags, read_jsonl, set_seeds
from src.featurize import featurize
from src.model import FastTextClassifier, batch_ids


def encode(rows, theme_ids, image_ids, n_buckets):
    """Turn rows into (bucket id lists, theme index tensor, image 0/1 matrix)."""
    ids = [featurize(r["text"], n_buckets) for r in rows]
    y_theme = torch.tensor([theme_ids.index(r["theme"]) for r in rows])
    y_img = torch.tensor([[float(i in r["images"]) for i in image_ids] for r in rows])
    return ids, y_theme, y_img


@torch.no_grad()
def evaluate(model, data, loss_fn):
    """Val loss + the metrics we log each epoch."""
    model.eval()
    ids, y_theme, y_img = data
    t_logits, i_logits = model(*batch_ids(ids))
    loss = loss_fn(t_logits, i_logits, y_theme, y_img).item()
    pred_t = t_logits.argmax(1)
    pred_i = (torch.sigmoid(i_logits) >= 0.5).numpy()  # 0.5 for now; step 6 calibrates this
    model.train()
    return {
        "loss": loss,
        "theme_acc": accuracy_score(y_theme, pred_t),
        # macro-F1: F1 per theme, then a plain average, so every theme
        # (including "none") counts equally regardless of how common it is.
        "theme_macroF1": f1_score(y_theme, pred_t, average="macro", zero_division=0),
        # micro-F1: pool every (request, image) decision together, then F1.
        "image_microF1": f1_score(y_img.numpy(), pred_i, average="micro", zero_division=0),
    }


def train_one(cfg, tags, train_rows, val_rows, log, verbose=True):
    set_seeds(cfg.get("seed", SEED))  # same seed -> same init + same batch order -> same result
    theme_ids = [t["id"] for t in tags["themes"]] + ["none"]  # head order: themes, then none
    image_ids = [i["id"] for i in tags["images"]]
    m = cfg["model"]
    tr = encode(train_rows, theme_ids, image_ids, m["buckets"])
    va = encode(val_rows, theme_ids, image_ids, m["buckets"])

    model = FastTextClassifier(m["buckets"], m["dim"], len(theme_ids), len(image_ids), m["hidden"])
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        log.info("model: B=%d d=%d hidden=%s -> %s parameters", m["buckets"], m["dim"], m["hidden"], f"{n_params:,}")

    # Images are rare (each is in ~10% of requests), so a model that always says
    # "no" would already be 90% right. pos_weight makes a missed image cost
    # (negatives/positives) times more than a false alarm, capped at 10.
    pos = tr[2].sum(0)
    neg = len(train_rows) - pos
    pos_weight = (neg / pos.clamp(min=1)).clamp(max=cfg["pos_weight_cap"])
    ce = nn.CrossEntropyLoss()  # softmax + -log(prob of the true theme)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # sigmoid + yes/no log loss, per image
    lam = cfg["image_loss_weight"]

    def loss_fn(t_logits, i_logits, y_t, y_i):
        return ce(t_logits, y_t) + lam * bce(i_logits, y_i)

    bs, max_epochs = cfg["batch_size"], cfg["max_epochs"]
    steps_per_epoch = (len(train_rows) + bs - 1) // bs
    if cfg["optim"]["name"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["optim"]["lr"])
        sched = None
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["optim"]["lr"])
        # fastText's schedule: lr falls in a straight line from lr to 0 over all
        # planned steps. Big steps early (move fast), tiny steps late (settle).
        total = steps_per_epoch * max_epochs
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: max(0.0, 1 - step / total))

    rng = random.Random(cfg.get("seed", SEED))
    order = list(range(len(train_rows)))
    best, best_state, best_epoch, bad_epochs, history = -1.0, None, 0, 0, []

    for epoch in range(1, max_epochs + 1):
        rng.shuffle(order)  # new batch order every epoch, but reproducible
        total_loss = 0.0
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            t_logits, i_logits = model(*batch_ids([tr[0][i] for i in idx]))
            loss = loss_fn(t_logits, i_logits, tr[1][idx], tr[2][idx])
            opt.zero_grad()   # clear gradients from the previous step
            # Backprop the batch SUM, not the mean. PyTorch losses average over
            # the batch, which divides every gradient by 32; fastText updates
            # after each single example, so its lr means "per example". Summing
            # makes lr 0.5 here mean what lr 0.5 means in fastText. Without this,
            # SGD barely moved (val acc stuck at 0.118 = always "none").
            # Adam is unaffected: it rescales step sizes by gradient magnitude.
            (loss * len(idx)).backward()   # compute new gradients
            opt.step()        # update weights
            if sched:
                sched.step()
            total_loss += loss.item() * len(idx)

        v = evaluate(model, va, loss_fn)
        v["epoch"], v["train_loss"] = epoch, total_loss / len(order)
        history.append(v)
        if verbose:
            log.info("epoch %2d | train loss %.4f | val loss %.4f | val theme acc %.3f | val theme macro-F1 %.3f | val image micro-F1 %.3f | lr %.4f",
                     epoch, v["train_loss"], v["loss"], v["theme_acc"], v["theme_macroF1"], v["image_microF1"], opt.param_groups[0]["lr"])

        # Selection score. "theme" (v1, the spec): val theme macro-F1 only.
        # "mean" (v2): average of theme macro-F1 and image micro-F1. With 100
        # images the image head learns slowly, and a theme-only score stopped
        # training while images were at F1 0.28 (see NOTES, v2 step 5).
        v["score"] = v["theme_macroF1"] if cfg.get("select_metric", "theme") == "theme" else (v["theme_macroF1"] + v["image_microF1"]) / 2
        if v["score"] > best:
            best, best_epoch, bad_epochs = v["score"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())  # snapshot the best weights
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                if verbose:
                    log.info("early stop: no val score gain for %d epochs (best %.3f at epoch %d)", bad_epochs, best, best_epoch)
                break

    return {"best_score": best, "best_f1": history[best_epoch - 1]["theme_macroF1"], "best_epoch": best_epoch, "state_dict": best_state, "history": history,
            "theme_ids": theme_ids, "image_ids": image_ids, "n_params": n_params,
            "best_metrics": history[best_epoch - 1]}


def save(result, cfg, log):
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "fasttext_best.pt"
    torch.save({"config": cfg, "state_dict": result["state_dict"], "theme_ids": result["theme_ids"],
                "image_ids": result["image_ids"], "best_epoch": result["best_epoch"]}, path)
    log.info("saved best checkpoint (epoch %d, val theme macro-F1 %.3f, score %.3f) to %s", result["best_epoch"], result["best_f1"], result["best_score"], path.relative_to(ARTIFACTS.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--adam", action="store_true", help="fallback: Adam with lr 1e-2")
    args = ap.parse_args()

    log = get_logger("train_sweep" if args.sweep else "train")
    torch.set_num_threads(4)
    with open(CONFIG / "train.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.adam:
        cfg["optim"] = {"name": "adam", "lr": 0.01}
    tags = load_tags()
    train_rows, val_rows = read_jsonl(DATA / "splits/train.jsonl"), read_jsonl(DATA / "splits/val.jsonl")
    log.info("train %d rows, val %d rows | config %s", len(train_rows), len(val_rows), {k: v for k, v in cfg.items() if k != "sweep"})

    if not args.sweep:
        save(train_one(cfg, tags, train_rows, val_rows, log), cfg, log)
        return

    # Sweep over whatever keys train.yaml lists. Each key maps to where it lives in the config.
    where = {"dim": ("model", "dim"), "buckets": ("model", "buckets"), "hidden": ("model", "hidden"),
             "lr": ("optim", "lr"), "pos_weight_cap": (None, "pos_weight_cap"),
             "image_loss_weight": (None, "image_loss_weight")}
    sw = cfg["sweep"]
    keys = list(sw)
    combos = list(itertools.product(*(sw[k] for k in keys)))
    log.info("sweep: %d runs over %s", len(combos), keys)
    rows, best = [], None
    for n, values in enumerate(combos, 1):
        c = copy.deepcopy(cfg)
        for k, v in zip(keys, values):
            section, field = where[k]
            (c[section] if section else c)[field] = v
        t0 = time.time()
        r = train_one(c, tags, train_rows, val_rows, log, verbose=False)
        bm, secs = r["best_metrics"], time.time() - t0
        rows.append((*values, r["n_params"], r["best_epoch"], bm["theme_acc"], r["best_f1"], bm["image_microF1"], r["best_score"], secs))
        log.info("[%2d/%d] %s | params %s | best epoch %2d | val theme acc %.3f macro-F1 %.3f | image micro-F1 %.3f | %.1fs",
                 n, len(combos), " ".join(f"{k}={v}" for k, v in zip(keys, values)), f"{r['n_params']:,}",
                 r["best_epoch"], bm["theme_acc"], r["best_f1"], bm["image_microF1"], secs)
        # Ties broken by fewer parameters: same score, smaller model wins.
        if best is None or (r["best_score"], -r["n_params"]) > (best[0]["best_score"], -best[0]["n_params"]):
            best = (r, c)

    cols = keys + ["params", "best epoch", "val theme acc", "val theme macro-F1", "val image micro-F1", "score", "sec"]
    table = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    f1_col = len(keys) + 5  # sort by the selection score
    for row in sorted(rows, key=lambda x: -x[f1_col]):
        vals = [str(v) for v in row[: len(keys)]] + [f"{row[len(keys)]:,}", str(row[len(keys) + 1])] + [f"{v:.3f}" for v in row[len(keys) + 2 : -1]] + [f"{row[-1]:.1f}"]
        table.append("| " + " | ".join(vals) + " |")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "sweep.md").write_text("# Sweep (sorted by val theme macro-F1)\n\n" + "\n".join(table) + "\n")
    log.info("sweep table written to reports/sweep.md\n%s", "\n".join(table))

    r, c = best
    log.info("BEST: %s lr=%s pos_weight_cap=%s image_loss_weight=%s (val theme macro-F1 %.3f, score %.3f)",
             c["model"], c["optim"]["lr"], c["pos_weight_cap"], c["image_loss_weight"], r["best_f1"], r["best_score"])
    log.info("per-epoch history of the best run:")
    for v in r["history"]:
        log.info("  epoch %2d | train loss %.4f | val loss %.4f | acc %.3f | macro-F1 %.3f | img micro-F1 %.3f",
                 v["epoch"], v["train_loss"], v["loss"], v["theme_acc"], v["theme_macroF1"], v["image_microF1"])
    save(r, c, log)


if __name__ == "__main__":
    main()
