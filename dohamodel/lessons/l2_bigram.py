"""Lesson 2: the bigram model, the dumbest possible language model.

It predicts the next character from the current character only. The whole model
is one table W of shape [V, V]: row = current char, column = score ("logit") for
each possible next char. softmax(row) turns scores into probabilities.

Run:  .venv/bin/python lessons/l2_bigram.py [weight_decay]   (default 0.0)
"""
import math
import random
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
random.seed(0)

text = unicodedata.normalize("NFC", Path(__file__).parent.parent.joinpath("data/toy_kabir.txt").read_text(encoding="utf-8"))
dohe = text.strip().split("\n\n")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
V = len(chars)

# --- split by doha, not by character, so validation dohe are truly unseen ---
random.shuffle(dohe)
n_val = round(0.15 * len(dohe))
val_dohe, train_dohe = dohe[:n_val], dohe[n_val:]

def to_pairs(dohe_list):
    """Every (current char, next char) pair. Dohe are separated by a blank line."""
    s = "\n\n".join(dohe_list) + "\n\n"
    ids = torch.tensor([stoi[c] for c in s])
    return ids[:-1], ids[1:]          # x[i] -> should predict y[i]

xtr, ytr = to_pairs(train_dohe)
xva, yva = to_pairs(val_dohe)
print(f"train: {len(train_dohe)} dohe, {len(xtr):,} pairs   val: {len(val_dohe)} dohe, {len(xva):,} pairs")

# --- 1. counting: the exact best bigram for the training set (no gradient descent) ---
counts = torch.zeros(V, V)
for a, b in zip(xtr.tolist(), ytr.tolist()):
    counts[a, b] += 1
probs = counts / counts.sum(1, keepdim=True).clamp(min=1)
floor = -probs[xtr, ytr].log().mean().item()
val_counts = -probs[xva, yva].log().mean().item()
print(f"\nbest possible bigram loss on TRAIN (by counting): {floor:.3f}")
print(f"same counted table on VAL: {val_counts:.3f}   <- inf means a val pair never appeared in train")

# --- 2. the same model, learned by gradient descent ---
W = torch.zeros(V, V, requires_grad=True)       # all zeros = every next char equally likely
wd = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
print(f"\nweight decay: {wd}")
opt = torch.optim.AdamW([W], lr=0.1, weight_decay=wd)
best_val = (float("inf"), 0)

for step in range(501):
    loss = F.cross_entropy(W[xtr], ytr)         # mean of -log p(true next char)
    with torch.no_grad():
        vloss = F.cross_entropy(W[xva], yva)
    best_val = min(best_val, (vloss.item(), step))
    if step % 100 == 0:
        print(f"step {step:3d}   train loss {loss.item():.3f}   val loss {vloss.item():.3f}")
    opt.zero_grad()
    loss.backward()                             # gradient of loss w.r.t. all V*V numbers
    opt.step()                                  # step every number slightly downhill

print(f"best val loss {best_val[0]:.3f} at step {best_val[1]}   (ln {V} = {math.log(V):.3f})")

# --- 3. look inside: what does the model expect after a space? after प्? ---
P = F.softmax(W.detach(), dim=1)
for ctx in [" ", "्", "क"]:
    top = P[stoi[ctx]].topk(6)
    shown = "  ".join(f"{itos[i]!r} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
    print(f"after {ctx!r}: {shown}")

# --- 4. generate: predict, pick, repeat ---
g = torch.Generator().manual_seed(42)
idx, out = stoi["\n"], []
for _ in range(160):
    idx = torch.multinomial(P[idx], 1, generator=g).item()
    out.append(itos[idx])
print("\nsample:\n" + "".join(out))
