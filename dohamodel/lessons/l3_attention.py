"""Lesson 3: attention, built up in four small steps (no training yet).

Each character position holds a vector of C numbers (its "embedding").
Attention lets position t build a new vector as a weighted average of the
vectors at positions 0..t (never t+1.., that would be seeing the future).

Run:  .venv/bin/python lessons/l3_attention.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False, linewidth=120)

T, C = 6, 8                      # 6 positions ("मन का फ"), 8 numbers per position
x = torch.randn(T, C)            # stand-in for 6 character embeddings

# --- A. plain averaging of the past, the slow way ---
slow = torch.stack([x[: t + 1].mean(0) for t in range(T)])

# --- A'. the same thing as ONE matrix multiply ---
# Row t of `wei` holds the weights position t gives to positions 0..T-1.
# Lower-triangular = can only look backwards; each row sums to 1 = an average.
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(1, keepdim=True)
fast = wei @ x
print("A. equal-weight averaging matrix (row = 'me', column = 'who I look at'):")
print(wei)
print("loop and matrix give the same answer:", torch.allclose(slow, fast))

# --- B. learned weights: query . key, masked, softmax ---
# Each position emits a query ("what am I looking for?") and a key ("what do I hold?").
# Score = query . key: big when they match. These two Linear layers are what training learns.
head = 4
query, key, value = (torch.nn.Linear(C, head, bias=False) for _ in range(3))
q, k, v = query(x), key(x), value(x)                     # each [T, head]
scores = q @ k.T / head**0.5                             # [T, T]; /sqrt(head) explained in C
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))        # future -> -inf -> softmax gives exactly 0
att = F.softmax(scores, dim=1)
out = att @ v                                            # mix the VALUES, using the learned weights
print("\nB. learned attention weights (random init, before any training):")
print(att)
print("every row sums to 1:", torch.allclose(att.sum(1), torch.ones(T)),
      "  upper triangle all zero:", bool((att.triu(1) == 0).all()))

# --- C. why divide by sqrt(head size)? ---
# A dot product of two random d-long vectors has variance ~d. Big scores -> softmax picks
# one position with ~100% weight and gradients for all the others vanish.
for d in (4, 64, 512):
    a, b = torch.randn(1000, d), torch.randn(1000, d)
    s = (a * b).sum(1)
    unscaled = F.softmax(torch.stack([s[i:i + 6] for i in range(0, 600, 6)]), dim=1).max(1).values.mean()
    scaled = F.softmax(torch.stack([s[i:i + 6] for i in range(0, 600, 6)]) / d**0.5, dim=1).max(1).values.mean()
    print(f"C. d={d:3d}: score std {s.std():5.1f}   avg top weight over 6 positions: unscaled {unscaled:.0%}, scaled {scaled:.0%}")

# --- D. attention alone cannot see order ---
# Shuffle positions 0..4 and look at the output of the last position (5): unchanged,
# because a weighted sum doesn't care what order its terms arrive in.
perm = torch.tensor([3, 0, 4, 1, 2, 5])
xp = x[perm]
out_p = F.softmax((query(xp) @ key(xp).T / head**0.5).masked_fill(~mask, float("-inf")), dim=1) @ value(xp)
print("\nD. last position, original order:", out[-1])
print("   last position, earlier chars shuffled:", out_p[-1])
print("   identical?", torch.allclose(out[-1], out_p[-1], atol=1e-6))

# Fix: add a learned "position embedding" to each character before attention.
pos = torch.randn(T, C)          # in the real model this is learned, one vector per position 0..127
out_pos = lambda z: F.softmax((query(z) @ key(z).T / head**0.5).masked_fill(~mask, float("-inf")), dim=1) @ value(z)
print("   with position embeddings added, identical?",
      torch.allclose(out_pos(x + pos)[-1], out_pos(xp + pos)[-1], atol=1e-6))
