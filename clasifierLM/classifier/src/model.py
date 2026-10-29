"""Step 5.1: the fastText-style classifier.

    bucket ids --> EmbeddingBag(B, d, mode="mean") --> h (d numbers)
                    [optional: Linear(d, d) + ReLU]
                 /                                \\
    theme head: Linear(d, T+1)            image head: Linear(d, I)
    softmax -> one winner (incl. none)    sigmoid -> each image yes/no

How it "understands" a request: every piece ("<lamp>", "lam", ...) owns one
row of d learned numbers. The request vector h is simply the AVERAGE of its
pieces' rows. Training nudges those rows so that, e.g., the pieces of
"funeral", "last breath" and "dying" all push h in the direction that the theme
head reads as mrityu. It's a linear model on top of learned word-piece vectors:
tiny, fast, and easy to reimplement in JavaScript.
"""

import torch
from torch import nn

from src.featurize import featurize


class FastTextClassifier(nn.Module):
    def __init__(self, n_buckets: int, dim: int, n_themes: int, n_images: int, hidden: bool = False):
        super().__init__()
        # EmbeddingBag = Embedding lookup + mean in one fast op. It takes ALL
        # requests' ids as one flat list plus "offsets" saying where each
        # request starts, so requests of different lengths batch without padding:
        #   ids = [5, 9, 2,   7, 7]   offsets = [0, 3]  -> request 0 = [5,9,2], request 1 = [7,7]
        self.emb = nn.EmbeddingBag(n_buckets, dim, mode="mean")
        self.hidden = nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) if hidden else nn.Identity()
        self.theme_head = nn.Linear(dim, n_themes)  # n_themes already includes "none"
        self.image_head = nn.Linear(dim, n_images)

        # fastText initialization: small random embeddings, all-zero heads.
        # Zero heads mean every class starts at equal probability; the
        # embeddings must be random (not zero) or every request would look identical.
        nn.init.uniform_(self.emb.weight, -1.0 / dim, 1.0 / dim)
        if hidden:
            # Deviation from the spec, hidden mode only. With ±1/d embeddings,
            # h (a mean of ~80 rows) is ~0.002, while the hidden layer's random
            # bias is ~±0.18. Every ReLU then sees only its bias: half are dead
            # for every request, the rest output the same value for every
            # request, and nothing learns (measured: 56–100% dead units, val
            # macro-F1 stuck at 0.019 for all lr). Bigger embeddings + zero
            # hidden bias let the request actually reach the ReLUs.
            # (Experiments in NOTES.md, step 5.)
            nn.init.uniform_(self.emb.weight, -1.0, 1.0)
            nn.init.zeros_(self.hidden[0].bias)
        for head in (self.theme_head, self.image_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor):
        h = self.hidden(self.emb(ids, offsets))
        # Return raw scores ("logits"). The losses apply softmax/sigmoid
        # internally, which is more numerically stable than doing it here.
        return self.theme_head(h), self.image_head(h)


def batch_ids(id_lists: list[list[int]]):
    """[[5,9,2], [7,7]] -> ids=tensor([5,9,2,7,7]), offsets=tensor([0,3])."""
    offsets, total = [], 0
    for ids in id_lists:
        offsets.append(total)
        total += len(ids)
    flat = [i for ids in id_lists for i in ids]
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class FastTextPredictor:
    """Wraps a saved checkpoint with the same predict() interface as the baselines."""

    def __init__(self, ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        self.cfg, self.theme_ids, self.image_ids = ckpt["config"], ckpt["theme_ids"], ckpt["image_ids"]
        m = self.cfg["model"]
        self.model = FastTextClassifier(m["buckets"], m["dim"], len(self.theme_ids), len(self.image_ids), m["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()  # inference mode (matters for layers like dropout; good habit)

    @torch.no_grad()  # no gradient bookkeeping needed for prediction
    def predict(self, texts):
        out = []
        for text in texts:
            ids = featurize(text, self.cfg["model"]["buckets"])
            if not ids:
                # Empty request (e.g. "?!" or all non-ASCII): nothing to average.
                # Spec: predict "none". (EmbeddingBag would return a zero vector,
                # giving whatever the biases prefer, so we decide explicitly.)
                out.append({"theme_probs": {t: float(t == "none") for t in self.theme_ids},
                            "image_probs": {i: 0.0 for i in self.image_ids}})
                continue
            t_logits, i_logits = self.model(*batch_ids([ids]))
            tp = torch.softmax(t_logits[0], dim=0).tolist()
            ip = torch.sigmoid(i_logits[0]).tolist()
            out.append({"theme_probs": dict(zip(self.theme_ids, tp)),
                        "image_probs": dict(zip(self.image_ids, ip))})
        return out
