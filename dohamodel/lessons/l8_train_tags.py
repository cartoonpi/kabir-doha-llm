"""Lesson 8: the tag-steered model (PRD build step 5), and one point on the data-scaling curve.

Same model.py as Lesson 7. What changes is the data and the format:

    <maya><rassi><kahekabir>|रस्सी देखी साँप सा, मूरख भरम न जाय।
    माया ठगिनी जग ठगे, कबिरा सत्य न पाय॥

  - each tag is ONE token (a new row in the embedding table), "|" separates tags from the doha
  - no loss on tags: a target that is a tag or "|" is set to -100, which cross_entropy skips.
    The model learns to USE tags, not to predict which tags are common.
  - every batch is half real, half synthetic (--real_frac), so 1,731 real dohe aren't drowned out
  - stage 1: random weights, real train + N synthetic dohe, early stopping on val loss (306 real val dohe)
    stage 2: start from stage 1's best weights, real dohe only, learning rate / 10 (pulls the style back to Kabir)

The vocabulary is the same for every N (all characters of real + all synthetic dohe, plus all tags), so
runs with different N are directly comparable. Val loss counts only doha characters (tags are input).

After stage 2, generates --samples dohe for random tag combinations and measures:
  shape, copy rate, rhythm (matra.grade: exact / near), image hit (requested image word appears, vs the
  rate for a random other image), device hit (sadho / prashn / kahekabir cue word appears).

Run:  .venv/bin/python -u lessons/l8_train_tags.py --synth 5000 --name tags_5k [--steps 6000] [--ft_steps 600]
"""
import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_prep"))
from matra import grade  # noqa: E402
from model import GPT, Config  # noqa: E402
from tag_real import IMAGE_WORDS, tag as detect  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--synth", type=int, default=5000, help="synthetic dohe to use (0 = real only, with tags)")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--ft_steps", type=int, default=600)
p.add_argument("--real_frac", type=float, default=0.5)
p.add_argument("--dropout", type=float, default=0.15)
p.add_argument("--n_embd", type=int, default=192)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=200)
p.add_argument("--samples", type=int, default=300)
p.add_argument("--name", default="tags")
p.add_argument("--device", default="cpu", help="cpu, or cuda (also AMD GPUs under ROCm PyTorch)")
p.add_argument("--bf16", action="store_true", help="bfloat16 autocast for the forward pass (GPU matrix units; weights stay float32)")
args = p.parse_args()
torch.manual_seed(0)

# ---------------- data ----------------
real = [json.loads(l) for l in (ROOT / "data/clean/kabir_real_tagged.jsonl").open(encoding="utf-8")]
synth_all = [json.loads(l) for l in (ROOT / "data/synth/full/kept.jsonl").open(encoding="utf-8")]
synth = random.Random(0).sample(synth_all, min(args.synth, len(synth_all)))
real_train = [r for r in real if r["split"] == "train"]
real_val = [r for r in real if r["split"] == "val"]

THEMES = ["guru", "maya", "mrityu", "ahankar", "pakhand", "bhakti", "vani", "dhairya", "satsang", "prem"]
IMAGES = list(IMAGE_WORDS)
DEVICES = ["ulatbansi", "sadho", "prashn", "kahekabir"]
TAGS = [f"<{t}>" for t in THEMES + IMAGES + DEVICES]
SEP = "|"
chars = sorted(set("".join(r["text"] for r in real + synth_all)) | {"\n"})
itos = chars + [SEP] + TAGS
stoi = {s: i for i, s in enumerate(itos)}
FIRST_TAG = stoi[SEP]                      # ids >= this are "|" or tags: never predicted


def tags_of(r):
    return ([f"<{r['theme']}>"] if r.get("theme") else []) + [f"<{i}>" for i in r.get("images", [])] + \
           ([f"<{r['device']}>"] if r.get("device") else [])


def encode(rows):
    """One stream: <tags>|doha\\n\\n<tags>|doha\\n\\n ...  as token ids."""
    ids = []
    for r in rows:
        ids += [stoi[t] for t in tags_of(r)] + [stoi[SEP]] + [stoi[c] for c in r["text"] + "\n\n"]
    return torch.tensor(ids)


real_ids, val_ids = encode(real_train), encode(real_val)
synth_ids = encode(synth) if synth else None

cfg = Config(vocab_size=len(itos), n_embd=args.n_embd, n_layer=args.n_layer, dropout=args.dropout)
model = GPT(cfg).to(args.device)
n_chars = len(real_ids) + (len(synth_ids) if synth_ids is not None else 0)
print(f"vocab {len(itos)} ({len(chars)} chars + | + {len(TAGS)} tags)   parameters {model.n_params():,}")
print(f"real train {len(real_train):,} dohe   synthetic {len(synth):,}   tokens {n_chars:,}   "
      f"params per token {model.n_params() / n_chars:.1f}   real share of each batch {args.real_frac if synth else 1:.0%}")


def windows(ids, n):
    s = torch.randint(len(ids) - cfg.block_size - 1, (n,))
    return (torch.stack([ids[i:i + cfg.block_size] for i in s]).to(args.device),
            torch.stack([ids[i + 1:i + cfg.block_size + 1] for i in s]).to(args.device))


def mask(y):
    y = y.clone()
    y[y >= FIRST_TAG] = -100
    return y


def get_batch(real_only=False, bs=64):
    if real_only or synth_ids is None:
        x, y = windows(real_ids, bs)
    else:
        k = round(bs * args.real_frac)
        (xr, yr), (xs, ys) = windows(real_ids, k), windows(synth_ids, bs - k)
        x, y = torch.cat([xr, xs]), torch.cat([yr, ys])
    return x, mask(y)


@torch.no_grad()
def eval_loss(ids, max_chunks=None):
    model.eval()
    tot = n = 0
    for s in list(range(0, len(ids) - 1, cfg.block_size))[:max_chunks]:
        x, y = ids[s:s + cfg.block_size][None].to(args.device), mask(ids[s + 1:s + cfg.block_size + 1][None]).to(args.device)
        _, loss = model(x[:, :y.size(1)], y)
        k = (y != -100).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / n


out = ROOT / "checkpoints" / args.name
out.mkdir(parents=True, exist_ok=True)
EVAL_CHUNKS = len(val_ids) // cfg.block_size + 1
log, t0 = [], time.time()


def train(stage, steps, lr, real_only):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    warm = 100 if stage == 1 else 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(warm, 1)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))))
    every = args.eval_every if stage == 1 else 50
    best = (float("inf"), 0)
    for step in range(steps + 1):
        if step % every == 0:
            tr, va = eval_loss(real_ids, EVAL_CHUNKS), eval_loss(val_ids)
            sy = eval_loss(synth_ids, EVAL_CHUNKS) if synth_ids is not None else None
            log.append({"stage": stage, "step": step, "train_real": tr, "train_synth": sy, "val": va})
            if va < best[0]:
                best = (va, step)
                torch.save(model.state_dict(), out / f"stage{stage}.pt")
            sy_s = f"   synth {sy:.3f}" if sy is not None else ""
            print(f"stage {stage}  step {step:5d}   real-train {tr:.3f}{sy_s}   val {va:.3f}   "
                  f"best {best[0]:.3f}@{best[1]}   {time.time() - t0:.0f}s", flush=True)
        if step == steps:
            break
        x, y = get_batch(real_only)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
            _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.load_state_dict(torch.load(out / f"stage{stage}.pt", map_location=args.device))
    return best


b1 = train(1, args.steps, args.lr, real_only=False)
print(f"\nstage 1 best val {b1[0]:.3f} at step {b1[1]}\n")
b2 = train(2, args.ft_steps, args.lr / 10, real_only=True)
print(f"\nstage 2 best val {b2[0]:.3f} at step {b2[1]}   (Lesson 7, real only, no tags: 1.660)")

# ---------------- generate for random tag combinations ----------------
real_words = [re.findall(r"\S+", r["text"].replace("\n", " ")) for r in real if r["split"] != "drop"]
grams = {tuple(w[i:i + 5]) for w in real_words for i in range(len(w) - 4)}
img_forms = {i: set(ws.split()) for i, ws in IMAGE_WORDS.items()}
rng = random.Random(1)
g = torch.Generator().manual_seed(1)
model.cpu().eval()                         # sampling is one character at a time: faster on the CPU
samples = []
for _ in range(args.samples):
    req = {"theme": rng.choice(THEMES), "images": rng.sample(IMAGES, rng.choice([1, 2])), "device": rng.choice(DEVICES)}
    prompt = torch.tensor([[stoi["\n"], stoi["\n"]] + [stoi[t] for t in tags_of(req)] + [stoi[SEP]]])
    ids = model.generate(prompt, 160, temperature=0.8, stop_id=stoi["॥"], generator=g)[0, prompt.size(1):].tolist()
    samples.append({**req, "text": "".join(itos[i] for i in ids if i < FIRST_TAG).strip()})


def has_image(text, img):
    return bool(img_forms[img] & set(re.findall(r"[ऀ-ॿ]+", text)))


n = len(samples)
shaped = sum(s["text"].count("\n") == 1 and s["text"].endswith("॥") for s in samples)
copies = sum(any(tuple(w[i:i + 5]) in grams for i in range(len(w) - 4))
             for w in (re.findall(r"\S+", s["text"].replace("\n", " ")) for s in samples))
grades = [grade(s["text"]) for s in samples]
img_hit = sum(has_image(s["text"], i) for s in samples for i in s["images"]) / sum(len(s["images"]) for s in samples)
img_base = sum(has_image(s["text"], rng.choice([i for i in IMAGES if i not in s["images"]])) for s in samples) / n
dev = [s for s in samples if s["device"] != "ulatbansi"]
dev_hit = sum(detect(s["text"])[1] == s["device"] for s in dev) / len(dev)
print(f"\n{n} samples (stage 2 model, temperature 0.8, random tags):")
print(f"  two-line doha shape: {shaped / n:.0%}")
print(f"  copy rate (5+ words from a real doha): {copies / n:.0%}")
print(f"  rhythm: exact {grades.count('exact') / n:.0%}   near {grades.count('near') / n:.0%}   (Lesson 7: exact 2%)")
print(f"  image hit: {img_hit:.0%} of requested images appear   (a random other image: {img_base:.0%})")
print(f"  device hit: {dev_hit:.0%}  (sadho / prashn / kahekabir cue word as requested; chance about 1/3)")
print("\nfirst 10:")
for s in samples[:10]:
    print(f"  [{s['theme']}/{'+'.join(s['images'])}/{s['device']}] {grades[samples.index(s)] or '     '}  "
          + s["text"].replace("\n", " / "))

(out / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
(out / "log.json").write_text(json.dumps({
    "config": vars(cfg), "args": vars(args), "log": log,
    "best": {"stage1": {"val": b1[0], "step": b1[1]}, "stage2": {"val": b2[0], "step": b2[1]}},
    "itos": itos, "metrics": {"shape": shaped / n, "copy": copies / n, "exact": grades.count("exact") / n,
                              "near": grades.count("near") / n, "image_hit": img_hit, "image_base": img_base,
                              "device_hit": dev_hit},
}, ensure_ascii=False, indent=1), encoding="utf-8")
