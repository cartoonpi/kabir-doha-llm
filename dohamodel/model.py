"""The doha model: a small character-level GPT, written from scratch.

Everything here was built piece by piece in lessons/l3_attention.py. Reference:
Karpathy's nanoGPT (model.py), simplified.

Shapes: B = batch size, T = positions (<= block_size), C = n_embd (width).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 63
    block_size: int = 128     # context length: how far back the model can see
    n_layer: int = 4          # stacked blocks
    n_head: int = 4           # attention heads per block
    n_embd: int = 192         # numbers describing each position
    dropout: float = 0.15


class CausalSelfAttention(nn.Module):
    """Several attention heads in parallel (Lesson 3 B, done for all heads at once)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)   # query, key, value in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)      # mix the heads' outputs
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask)                              # saved with the model, not trained
        self.last_att = None                                            # kept for inspection in lessons

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_size]: each head gets its own slice of the width
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)            # [B, nh, T, T]
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))        # no peeking at the future
        att = F.softmax(att, dim=-1)
        self.last_att = att.detach()
        y = self.drop(att) @ v                                          # [B, nh, T, hs]
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # glue heads back together
        return self.drop(self.proj(y))


class FeedForward(nn.Module):
    """Per-position thinking: after gathering info with attention, process it (4x wider inside)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """attention then feed-forward, each wrapped as  x = x + f(layernorm(x))  (residual)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn, self.ff = CausalSelfAttention(cfg), FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # one vector per character
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # one vector per position (Lesson 3 D)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # vector -> score per next char
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # small random starting weights, so step-0 predictions are ~uniform (loss ~ ln vocab)
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                           # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=1.0, stop_id=None, generator=None):
        """Predict, pick, repeat (primer point 6). idx: [1, T] start ids."""
        for _ in range(max_new):
            logits, _ = self(idx[:, -self.cfg.block_size:])       # only the last 128 fit in context
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_id is not None and nxt.item() == stop_id:
                break
        return idx
