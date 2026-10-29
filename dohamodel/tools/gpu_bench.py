"""Time training steps of our model (Lesson 8 size: vocab 109, 4 x 192, batch 64 x 128) on CPU and GPU.

ROCm PyTorch exposes an AMD GPU through the torch.cuda API. The Radeon 840M (gfx1152) is not on ROCm's
official list, so run this once plain and once with HSA_OVERRIDE_GFX_VERSION=11.5.1 (pose as gfx1151).

Run:  .venv-rocm/bin/python tools/gpu_bench.py [--steps 30]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from model import GPT, Config  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=30)
p.add_argument("--n_embd", type=int, default=192)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--bf16", action="store_true", help="also time the GPU with bfloat16 autocast")
args = p.parse_args()

print(f"torch {torch.__version__}   hip {torch.version.hip}   gpu available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)}   arch: {torch.cuda.get_device_properties(0).gcnArchName}")


def bench(device, bf16=False):
    torch.manual_seed(0)
    model = GPT(Config(vocab_size=109, n_embd=args.n_embd, n_layer=args.n_layer)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    x = torch.randint(0, 109, (64, 128), device=device)
    y = torch.randint(0, 109, (64, 128), device=device)
    losses = []
    for i in range(args.steps + 5):
        if i == 5:                                   # first steps include kernel compilation: not timed
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
            _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())                   # .item() also waits for the GPU
    dt = (time.time() - t0) / args.steps
    return dt, losses[0], losses[-1]


print(f"model: {args.n_layer} layers x {args.n_embd} wide, "
      f"{GPT(Config(vocab_size=109, n_embd=args.n_embd, n_layer=args.n_layer)).n_params():,} parameters")
cpu = bench("cpu")
print(f"cpu:  {cpu[0]:.3f} s/step   loss {cpu[1]:.3f} -> {cpu[2]:.3f}")
if torch.cuda.is_available():
    try:
        gpu = bench("cuda")
        print(f"gpu:  {gpu[0]:.3f} s/step   loss {gpu[1]:.3f} -> {gpu[2]:.3f}   speedup {cpu[0] / gpu[0]:.1f}x")
        if args.bf16:
            g16 = bench("cuda", bf16=True)
            print(f"gpu bf16: {g16[0]:.3f} s/step   loss {g16[1]:.3f} -> {g16[2]:.3f}   speedup {cpu[0] / g16[0]:.1f}x")
    except Exception as e:                           # unsupported arch usually fails here, at the first kernel
        print("gpu failed:", type(e).__name__, str(e)[:300])
