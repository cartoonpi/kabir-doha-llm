"""Lesson 4: train the full transformer on the toy dohe and compare with the bigram.

Same train/val split as lessons/l2_bigram.py (same seed), so val numbers compare directly.

Run:  .venv/bin/python lessons/l4_train_toy.py [--steps 1500] [--dropout 0.15] [--n_embd 192] [--n_layer 4]
      add  --name big --snapshots 50,100,500  to save checkpoints/<name>/step_<N>.pt + log.json (for the web page)
"""
import argparse
import json
import math
import random
import sys
import time
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from model import GPT, Config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--dropout", type=float, default=0.15)
p.add_argument("--n_embd", type=int, default=192)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--name", default=None, help="save snapshots under checkpoints/<name>/")
p.add_argument("--snapshots", default="", help="comma-separated steps to save, e.g. 50,100,500")
args = p.parse_args()

torch.manual_seed(0)
random.seed(0)

# --- data: identical to Lesson 2 ---
text = unicodedata.normalize("NFC", (ROOT / "data/toy_kabir.txt").read_text(encoding="utf-8"))
dohe = text.strip().split("\n\n")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
random.shuffle(dohe)
n_val = round(0.15 * len(dohe))
val_dohe, train_dohe = dohe[:n_val], dohe[n_val:]
enc = lambda s: torch.tensor([stoi[c] for c in s])
train_ids = enc("\n\n".join(train_dohe) + "\n\n")
val_ids = enc("\n\n".join(val_dohe) + "\n\n")

cfg = Config(vocab_size=len(chars), n_embd=args.n_embd, n_layer=args.n_layer, dropout=args.dropout)
model = GPT(cfg)
print(f"parameters: {model.n_params():,}   train chars: {len(train_ids):,}   "
      f"parameters per train char: {model.n_params() / len(train_ids):.0f}")

def get_batch(batch_size=64):
    """64 random windows of 128 chars from the train text; target = the same window shifted by one."""
    starts = torch.randint(len(train_ids) - cfg.block_size - 1, (batch_size,))
    x = torch.stack([train_ids[s : s + cfg.block_size] for s in starts])
    y = torch.stack([train_ids[s + 1 : s + cfg.block_size + 1] for s in starts])
    return x, y

@torch.no_grad()
def eval_loss(ids):
    """Average loss over the whole text, in 128-char chunks. eval() switches dropout off."""
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(ids) - 1, cfg.block_size):
        x = ids[s : s + cfg.block_size][None]
        y = ids[s + 1 : s + cfg.block_size + 1][None]
        x = x[:, : y.size(1)]
        _, loss = model(x, y)
        total += loss.item() * y.numel()
        n += y.numel()
    model.train()
    return total / n

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
# learning rate: short warm-up, then cosine decay to 10% (the PRD's "1e-3 with decay")
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))

snap_steps = {int(s) for s in args.snapshots.split(",") if s}
snap_dir = ROOT / "checkpoints" / args.name if args.name else None
if snap_dir:
    snap_dir.mkdir(parents=True, exist_ok=True)

best = (float("inf"), 0)
t0 = time.time()
tr0, va0 = eval_loss(train_ids), eval_loss(val_ids)
log = [{"step": 0, "train": tr0, "val": va0}]
print(f"step    0   train {tr0:.3f}   val {va0:.3f}   (ln {cfg.vocab_size} = {math.log(cfg.vocab_size):.3f})")
for step in range(1, args.steps + 1):
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guard against occasional huge steps
    opt.step()
    sched.step()
    if step % 50 == 0:
        tr, va = eval_loss(train_ids), eval_loss(val_ids)
        log.append({"step": step, "train": tr, "val": va})
        if va < best[0]:
            best = (va, step)
            torch.save(model.state_dict(), ROOT / "lessons/l4_best.pt")
        if step % 100 == 0:
            print(f"step {step:4d}   train {tr:.3f}   val {va:.3f}   best val {best[0]:.3f}@{best[1]}   {time.time() - t0:.0f}s")

    if snap_dir and step in snap_steps:
        torch.save(model.state_dict(), snap_dir / f"step_{step}.pt")

print(f"\nbest val {best[0]:.3f} at step {best[1]}   (bigram best val: 2.736)")
if snap_dir:
    (snap_dir / "log.json").write_text(json.dumps({
        "config": vars(cfg), "args": vars(args), "log": log, "best": {"val": best[0], "step": best[1]},
        "snapshots": sorted(snap_steps), "itos": [itos[i] for i in range(len(itos))],
        "train_dohe": train_dohe, "val_dohe": val_dohe,
    }, ensure_ascii=False, indent=1))

# --- generate from the FINAL model and from the BEST-VAL model ---
def sample(prompt, n=3, temperature=0.8):
    g = torch.Generator().manual_seed(1)
    start = enc(prompt)[None]
    model.eval()
    for _ in range(n):
        out = model.generate(start, 100, temperature=temperature, stop_id=stoi["॥"], generator=g)
        print("  " + "".join(itos[i] for i in out[0].tolist()).replace("\n", " / "))

for label in ["final", "best-val"]:
    if label == "best-val":
        model.load_state_dict(torch.load(ROOT / "lessons/l4_best.pt"))
    print(f"\n[{label} model] prompt 'बुरा जो देखन' (a TRAIN doha):")
    sample("बुरा जो देखन")
    print(f"[{label} model] prompt from a blank line (free writing):")
    sample("\n")
