"""Step 4: two baselines that the neural model has to beat.

A baseline answers "is the fancy model worth it?". If a hand-written keyword
list scores as well as the neural net, the neural net isn't earning its keep.

Every model here exposes the same interface, which evaluate.py relies on:
    model.predict(texts) -> list of {"theme_probs": {id: p}, "image_probs": {id: p}}
theme_probs includes "none" and sums to 1. image_probs are independent (each 0..1).

Quick check (scores on val + human):   python -m src.baselines
"""

import re

import numpy as np
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from src.common import CONFIG, SEED, load_tags
from src.featurize import words


class KeywordModel:
    """Theme = the theme with the most keyword hits ("none" if zero hits).
    Images = every image with at least one hit."""

    def __init__(self, tags: dict):
        with open(CONFIG / "keywords.yaml", encoding="utf-8") as f:
            kw = yaml.safe_load(f)
        self.theme_ids = [t["id"] for t in tags["themes"]]
        self.image_ids = [i["id"] for i in tags["images"]]
        # One regex per tag: \b(kw1|kw2|...)s?\b on the normalized text.
        # Normalizing keywords the same way as the text ("show off" -> "show off")
        # means punctuation in either one can't break a match.
        def pattern(keywords):
            alts = "|".join(re.escape(" ".join(words(k))) for k in keywords)
            return re.compile(rf"\b(?:{alts})s?\b")
        self.theme_re = {t: pattern(kw["themes"][t]) for t in self.theme_ids}
        self.image_re = {i: pattern(kw["images"][i]) for i in self.image_ids}

    def predict(self, texts):
        out = []
        for text in texts:
            norm = " ".join(words(text))
            hits = {t: len(self.theme_re[t].findall(norm)) for t in self.theme_ids}
            total = sum(hits.values())
            if total == 0:
                theme_probs = {t: 0.0 for t in self.theme_ids} | {"none": 1.0}
            else:
                # Hit share as a pseudo-probability, so top-2 accuracy can be
                # computed. Ties go to the earlier theme in tags.yaml (dicts keep
                # order and max() returns the first maximum).
                theme_probs = {t: h / total for t, h in hits.items()} | {"none": 0.0}
            image_probs = {i: float(bool(self.image_re[i].search(norm))) for i in self.image_ids}
            out.append({"theme_probs": theme_probs, "image_probs": image_probs})
        return out


class LogRegModel:
    """Bag-of-words logistic regression.

    CountVectorizer turns each text into word counts ("the lamp lamp" ->
    {the: 1, lamp: 2}). Theme: one multinomial LogisticRegression over all
    themes + "none" (softmax, so probabilities sum to 1). Images: one
    independent yes/no LogisticRegression per image ("one-vs-rest").
    """

    def __init__(self, tags: dict):
        self.image_ids = [i["id"] for i in tags["images"]]
        # Same tokenization as everywhere else, so the comparison is fair.
        self.vec = CountVectorizer(tokenizer=words, lowercase=False, token_pattern=None)
        self.theme_clf = LogisticRegression(max_iter=2000, random_state=SEED)
        self.mlb = MultiLabelBinarizer(classes=self.image_ids)  # ["deepak"] -> [1,0,0,...]
        self.image_clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, random_state=SEED))

    def fit(self, rows):
        X = self.vec.fit_transform([r["text"] for r in rows])  # learn vocabulary from TRAIN only
        self.theme_clf.fit(X, [r["theme"] for r in rows])
        self.image_clf.fit(X, self.mlb.fit_transform([r["images"] for r in rows]))
        return self

    def predict(self, texts):
        X = self.vec.transform(texts)  # unknown words are silently ignored
        tp = self.theme_clf.predict_proba(X)
        ip = self.image_clf.predict_proba(X)
        return [
            {"theme_probs": dict(zip(self.theme_clf.classes_, map(float, t))),
             "image_probs": dict(zip(self.image_ids, map(float, i)))}
            for t, i in zip(tp, ip)
        ]


def quick_scores(model, rows, t_img=0.5):
    """Lightweight metrics for a sanity check; the full report is evaluate.py."""
    from sklearn.metrics import accuracy_score, f1_score

    preds = model.predict([r["text"] for r in rows])
    true_t = [r["theme"] for r in rows]
    pred_t = [max(p["theme_probs"], key=p["theme_probs"].get) for p in preds]
    image_ids = list(preds[0]["image_probs"])
    Y = np.array([[i in r["images"] for i in image_ids] for r in rows])
    P = np.array([[p["image_probs"][i] >= t_img for i in image_ids] for p in preds])
    return {
        "theme_acc": accuracy_score(true_t, pred_t),
        "theme_macroF1": f1_score(true_t, pred_t, average="macro", zero_division=0),
        "image_microF1": f1_score(Y, P, average="micro", zero_division=0),
    }


if __name__ == "__main__":
    from src.common import DATA, get_logger, read_jsonl

    log = get_logger("baselines")
    tags = load_tags()
    train = read_jsonl(DATA / "splits/train.jsonl")
    sets = {"val": read_jsonl(DATA / "splits/val.jsonl"), "human": read_jsonl(DATA / "requests_human.jsonl")}

    models = {"keyword": KeywordModel(tags), "logreg": LogRegModel(tags).fit(train)}
    for mname, m in models.items():
        for sname, rows in sets.items():
            s = quick_scores(m, rows)
            log.info("%-8s %-6s n=%-4d theme acc %.3f | theme macro-F1 %.3f | image micro-F1 %.3f",
                     mname, sname, len(rows), s["theme_acc"], s["theme_macroF1"], s["image_microF1"])

    # Peek inside the logistic regression: its top words per theme.
    lr = models["logreg"]
    vocab = np.array(lr.vec.get_feature_names_out())
    log.info("logreg vocabulary size: %d words", len(vocab))
    for cls, coefs in zip(lr.theme_clf.classes_, lr.theme_clf.coef_):
        log.info("  top words for %-8s %s", cls, ", ".join(vocab[np.argsort(coefs)[::-1][:8]]))
