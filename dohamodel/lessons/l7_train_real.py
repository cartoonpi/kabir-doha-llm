"""Lesson 7: the real-data baseline (PRD build step 1).

Same transformer as Lesson 4, now on 1,731 real Kabir dohe (data/clean/kabir_real.jsonl),
no tags yet. Measures the PRD's numbers at the best-val checkpoint:
  - held-out loss on the 306 val dohe
  - copy rate:   generated dohe sharing a run of 5+ words with any real doha
  - rhythm pass: generated dohe that scan 13+11 under matra.check(nasal, flex)
                 (real Kabir: 61%, see lessons/l6b_tune.py)

Saves snapshots to checkpoints/<name>/ in the same format as Lesson 4, so export.py and the
playground can show them.

Run:  .venv/bin/python -u lessons/l7_train_real.py [--steps 3000] [--n_embd 192] [--n_layer 4] [--name real]
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from matra import check  # noqa: E402
from model import GPT, Config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--dropout", type=float, default=0.15)
p.add_argument("--n_embd", type=int, default=192)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--name", default="real")
p.add_argument("--snapshots", default="250,500,1000,1500,2000,3000")
p.add_argument("--samples", type=int, default=200)
args = p.parse_args()
torch.manual_seed(0)

rows = [json.loads(l) for l in (ROOT / "data/clean/kabir_real.jsonl").open(encoding="utf-8")]
train_dohe = [r["text"] for r in rows if r["split"] == "train"]
val_dohe = [r["text"] for r in rows if r["split"] == "val"]
chars = sorted(set("\n\n".join(train_dohe + val_dohe)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
enc = lambda s: torch.tensor([stoi[c] for c in s])
train_ids = enc("\n\n".join(train_dohe) + "\n\n")
val_ids = enc("\n\n".join(val_dohe) + "\n\n")

cfg = Config(vocab_size=len(chars), n_embd=args.n_embd, n_layer=args.n_layer, dropout=args.dropout)
model = GPT(cfg)
print(f"vocab {len(chars)}   parameters {model.n_params():,}   train chars {len(train_ids):,}   "
      f"val chars {len(val_ids):,}   params per train char {model.n_params() / len(train_ids):.1f}")


def get_batch(bs=64):
    s = torch.randint(len(train_ids) - cfg.block_size - 1, (bs,))
    return (torch.stack([train_ids[i:i + cfg.block_size] for i in s]),
            torch.stack([train_ids[i + 1:i + cfg.block_size + 1] for i in s]))


@torch.no_grad()
def eval_loss(ids, max_chunks=None):
    model.eval()
    tot = n = 0
    starts = list(range(0, len(ids) - 1, cfg.block_size))[:max_chunks]
    for s in starts:
        x, y = ids[s:s + cfg.block_size][None], ids[s + 1:s + cfg.block_size + 1][None]
        _, loss = model(x[:, :y.size(1)], y)
        tot += loss.item() * y.numel()
        n += y.numel()
    model.train()
    return tot / n


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))
snap_steps = {int(s) for s in args.snapshots.split(",") if s}
out = ROOT / "checkpoints" / args.name
out.mkdir(parents=True, exist_ok=True)

TRAIN_EVAL_CHUNKS = len(val_ids) // cfg.block_size + 1      # same amount of text as val, for a fair comparison
best, log, t0 = (float("inf"), 0), [], time.time()
for step in range(0, args.steps + 1):
    if step % 100 == 0:
        tr, va = eval_loss(train_ids, TRAIN_EVAL_CHUNKS), eval_loss(val_ids)
        log.append({"step": step, "train": tr, "val": va})
        if va < best[0]:
            best = (va, step)
            torch.save(model.state_dict(), out / "best.pt")
        print(f"step {step:5d}   train {tr:.3f}   val {va:.3f}   best {best[0]:.3f}@{best[1]}   {time.time() - t0:.0f}s")
    if step in snap_steps:
        torch.save(model.state_dict(), out / f"step_{step}.pt")
    if step == args.steps:
        break
    x, y = get_batch()
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()

# the best-val checkpoint becomes a snapshot too
model.load_state_dict(torch.load(out / "best.pt"))
torch.save(model.state_dict(), out / f"step_{best[1]}.pt")
snap_steps.add(best[1])
print(f"\nbest val loss {best[0]:.3f} at step {best[1]}   (toy best 2.69, bigram 2.74 -- but a different val set)")

# --- PRD metrics on generated dohe from the best checkpoint ---
real_text = "\n".join(train_dohe + val_dohe)
real_words = [re.findall(r"\S+", d.replace("\n", " ")) for d in train_dohe + val_dohe]
grams = {tuple(w[i:i + 5]) for w in real_words for i in range(len(w) - 4)}


def is_copy(d):
    w = re.findall(r"\S+", d.replace("\n", " "))
    return any(tuple(w[i:i + 5]) in grams for i in range(len(w) - 4))


g = torch.Generator().manual_seed(1)
model.eval()
start = enc("\n\n")[None]
samples = []
for _ in range(args.samples):
    ids = model.generate(start, 160, temperature=0.8, stop_id=stoi["॥"], generator=g)[0, 2:].tolist()
    samples.append("".join(itos[i] for i in ids).strip())
shaped = [d for d in samples if d.count("\n") == 1 and d.endswith("॥")]
copies = sum(is_copy(d) for d in samples)
rhythm = sum(check(d, nasal=True, flex=True) for d in samples)
exact = sum(d in real_text for d in samples)
print(f"{len(samples)} samples at temperature 0.8:")
print(f"  two-line doha shape: {len(shaped) / len(samples):.0%}")
print(f"  copy rate (5+ words in a row from a real doha): {copies / len(samples):.0%}   exact copies: {exact / len(samples):.0%}")
print(f"  rhythm pass (13+11): {rhythm / len(samples):.0%}   (real Kabir: 61%)")
print("\nfirst 10 samples:")
for d in samples[:10]:
    tag = ("COPY " if is_copy(d) else "     ") + ("13+11 " if check(d, nasal=True, flex=True) else "      ")
    print(f"  {tag} {d.replace(chr(10), ' / ')}")

(out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "log.json").write_text(json.dumps({
    "config": vars(cfg), "args": vars(args), "log": log, "best": {"val": best[0], "step": best[1]},
    "snapshots": sorted(snap_steps), "itos": [itos[i] for i in range(len(itos))],
    "train_dohe": train_dohe, "val_dohe": val_dohe,
}, ensure_ascii=False, indent=1), encoding="utf-8")
