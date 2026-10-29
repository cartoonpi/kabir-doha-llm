# Task: Build the tag classifier (English request → doha tags)

You are building one component of a learning project: a tiny system that turns an English request ("something about how life runs out, use a lamp") into tags, which a separate character-level model then turns into a Kabir-style Hindi doha. **This spec covers only the tag classifier.** Do not touch the doha model.

The owner is learning ML. Prefer clear, commented, readable code over clever code. Explain non-obvious choices in comments.

---

## 0. Ground rules

- **Environment:** Linux, Python 3.11+, PyTorch (CPU is fine), numpy, scikit-learn, pyyaml, pytest. Node 18+ for the browser parity test. No GPU needed.
- **No pretrained models or embeddings.** Everything in the classifier is trained from scratch. The official `fasttext` pip package may be used *only* as a reference baseline in evaluation, never in the shipped model.
- **Reproducibility:** fix seeds (python, numpy, torch). Every script must be re-runnable and produce the same splits.
- **API cost guard:** never call the external LLM API without first printing the planned number of calls and an estimated token count, and asking the user to confirm. Support `--dry-run` everywhere the API is used.
- **Ask before:** changing the tag vocabulary, changing the hash function, or adding dependencies not listed here.
- **Work incrementally:** finish and test each step below before starting the next. After each step, append a short entry to `NOTES.md` (what was built, key numbers, open issues).

---

## 1. Repo layout

```
classifier/
  config/
    tags.yaml            # tag vocabulary (themes, images); single source of truth
    keywords.yaml        # keyword baseline: tag -> list of English keywords
    train.yaml           # hyperparameters
  data/
    requests_synth.jsonl # generated English requests (step 2)
    requests_human.jsonl # ~30 requests written by the owner by hand (never trained on)
    splits/              # train.jsonl, val.jsonl, test.jsonl
  src/
    llm_client.py        # thin wrapper around the external LLM API
    gen_requests.py      # step 2
    split_data.py        # step 2
    featurize.py         # step 3: text -> bucket ids (must match JS exactly)
    baselines.py         # step 4
    model.py             # step 5
    train.py             # step 5
    calibrate.py         # step 6
    evaluate.py          # step 7
    export.py            # step 8
    predict.py           # CLI
  web/
    classifier.js        # step 8: browser inference, no dependencies
    parity_test.mjs      # step 8
  tests/
  reports/
  NOTES.md
```

---

## 2. Training data: English requests

### 2.1 Tag vocabulary (`config/tags.yaml`)

Start with this placeholder; the owner will finalize it. Every tag has an id (used in the doha model too), an English name, and a one-line description that goes into generation prompts.

```yaml
themes:   # exactly one per request, or "none"
  - {id: guru,      name: guru,               desc: "the teacher who shows the way"}
  - {id: maya,      name: illusion,           desc: "worldly attachment, things that seem real but are not"}
  - {id: mrityu,    name: death,              desc: "mortality, life running out"}
  - {id: ahankar,   name: ego,                desc: "pride, arrogance, self-importance"}
  - {id: pakhand,   name: ritual hypocrisy,   desc: "empty rituals, pilgrimages and prayer beads without sincerity"}
  - {id: bhakti,    name: true devotion,      desc: "love of God found within, not outside"}
  - {id: vani,      name: speech,             desc: "the power of words, speaking sweetly"}
  - {id: dhairya,   name: patience,           desc: "waiting, things happen in their own time"}
  - {id: satsang,   name: good company,       desc: "the people you keep shape you"}
  - {id: prem,      name: love,               desc: "love that cannot be learned from books"}
images:   # zero, one or two per request
  - {id: deepak,  name: lamp}
  - {id: kumhar,  name: potter and clay}
  - {id: kuan,    name: well}
  - {id: machhli, name: fish}
  - {id: mala,    name: prayer beads}
  - {id: chandan, name: sandalwood}
  - {id: patang,  name: moth}
  - {id: bazaar,  name: market}
  - {id: naav,    name: boat}
  - {id: vriksh,  name: tree}
  # owner will extend to ~30
```

### 2.2 Generate requests (`src/gen_requests.py`)

`llm_client.py` exposes one function, `complete(prompt: str) -> str`. The provider and model come from env vars (`LLM_PROVIDER`, `LLM_MODEL`, API key var). **Ask the user which provider to implement first.**

Generation plan:

1. For each theme, sample about 150 combinations of (theme, 0–2 random images). About 30% have zero images.
2. For each combination, ask for 5 English requests in one call. Prompt requirements:
   - Write as a user typing into a doha generator would. Mix the registers: casual, poetic, indirect/personal ("my grandfather's last days"), very short ("death + lamp"), and long.
   - Mention or clearly imply each listed image. Never name an image that is not listed.
   - Do not use the theme's English name in more than about a third of the requests, so the model can't rely on one keyword.
   - Return JSON only: `[{"text": "..."}]`.
3. Add about 10% **"none"** requests that match no theme (cricket, tax filing, a cat video), with zero images.
4. Each output line: `{"text": str, "theme": theme_id | "none", "images": [image_id, ...], "source": "synth"}`.
5. Post-process:
   - Drop requests shorter than 2 words or longer than 40 words.
   - Drop exact duplicates after lowercasing.
   - Drop near-duplicates (word Jaccard similarity above 0.8 against any kept request).
   - Print counts per theme and per image.

Target: about 1,500–2,500 requests after cleaning. Run a pilot of 1 theme first (`--pilot`), show the user 20 random samples, and wait for approval before the full run.

### 2.3 Splits (`src/split_data.py`)

- 80/10/10 train/val/test, **stratified by theme**, fixed seed.
- `requests_human.jsonl` is a separate, never-trained test set. Create the file with the schema and 3 example lines, then tell the user to write about 30 of their own.

---

## 3. Featurization (`src/featurize.py`)

This must be **byte-for-byte reproducible in JavaScript**, so keep it simple and explicit.

1. Lowercase. Replace every character that is not `a-z`, `0-9` or a space with a space. Collapse whitespace. Split on spaces.
2. For each word `w`:
   - Whole-word piece: `"<" + w + ">"`.
   - Character 3-grams of `"<" + w + ">"` (for `cat`: `<ca`, `cat`, `at>`). Skip 3-grams if the word is 1 character.
3. Hash each piece with **32-bit FNV-1a over UTF-8 bytes**, then `bucket = hash % B`, with `B = 16384`.
   - Do **not** use Python's built-in `hash()`; it changes between runs.
   - FNV-1a: `h = 2166136261; for byte in data: h ^= byte; h = (h * 16777619) & 0xFFFFFFFF`.
4. Return the list of bucket ids for the whole request. Duplicates are kept, since they count for more in the average.

Tests (`tests/test_featurize.py`):
- Known FNV-1a values for `""`, `"a"`, `"<cat>"`; compute expected values independently in the test.
- Piece lists for `"cat"`, `"a"`, and a string full of punctuation.
- An empty request returns an empty list; the model must handle that by predicting "none".

---

## 4. Baselines (`src/baselines.py`)

Both are evaluated by the same `evaluate.py` as the main model.

1. **Keyword baseline:** `keywords.yaml` maps each tag to English keywords. Theme = the theme with the most keyword hits ("none" if zero). Images = every image with at least one hit. Seed the file with 5–10 obvious keywords per tag and tell the user to extend it.
2. **Logistic regression:** scikit-learn `LogisticRegression` on word counts (`CountVectorizer`) for the theme, and one-vs-rest logistic regression for images. About 20 lines of code.

---

## 5. Main model: fastText-style classifier

### 5.1 Architecture (`src/model.py`)

```
bucket ids ─► EmbeddingBag(B=16384, d=32, mode="mean") ─► h (32 numbers)
                                                      │
                     [optional hidden layer: Linear(32,32) + ReLU, off by default]
                                                      │
                        ┌─────────────────────────────┴────────────────────────┐
              theme head: Linear(32, n_themes + 1)               image head: Linear(32, n_images)
              softmax, one winner (incl. "none")                 sigmoid per image, independent
```

- Use `nn.EmbeddingBag` with `offsets`, so variable-length requests batch efficiently.
- Init: embeddings uniform in ±1/d. Both head weights and biases at zero (the fastText convention).
- Parameter count is about 16384×32 + 33×(11+30), roughly 0.53M. Print it at startup.

### 5.2 Training (`src/train.py`)

- Loss = `CrossEntropyLoss(theme) + λ · BCEWithLogitsLoss(images)`, with λ = 1.0 in `train.yaml`.
- For images, set `pos_weight` per image = (negatives / positives) from the training set, capped at 10, because most images are absent from most requests.
- Optimizer: plain SGD, lr 0.5, decaying linearly to 0 over training (fastText's default schedule). Fallback flag: Adam lr 1e-2.
- Batch size 32, up to 25 epochs. Early stopping on val theme macro-F1, patience 5. Save the best checkpoint.
- Log per epoch: train loss, val loss, val theme accuracy, val theme macro-F1, val image micro-F1.
- Small sweep in `train.yaml` (run sequentially; each takes seconds): d ∈ {16, 32, 64}, B ∈ {4096, 16384}, hidden ∈ {off, on}, lr ∈ {0.1, 0.5}. Choose by val theme macro-F1 and record the table in `NOTES.md`.

---

## 6. Calibration (`src/calibrate.py`)

On the **val** set only:

- **Theme "no clear match" threshold `t_theme`:** if the top theme probability is below `t_theme`, output "none". Sweep 0.2–0.8 and choose the value maximizing macro-F1, with "none" counted as a class.
- **Image threshold `t_img`:** keep an image if its sigmoid output is at least `t_img`, at most 2 images. Sweep 0.3–0.8 and maximize image micro-F1.
- Save both values to `artifacts/thresholds.json`.

---

## 7. Evaluation (`src/evaluate.py`)

Run every model (keyword, logistic regression, fastText-style, and optionally the official `fasttext` library as a reference) on **test** and **human** sets, applying the calibrated thresholds. Write `reports/classifier_eval.md` containing:

- **Theme:** top-1 accuracy, top-2 accuracy, macro-F1, precision and recall for "none".
- **Images:** micro-F1, and the exact-set match rate.
- The confusion matrix for themes (fastText-style model, test set).
- The 15 worst errors on the human set: request text, true tags, predicted tags with probabilities.
- A one-paragraph plain-English summary: which model wins, where it fails, what to try next.

Acceptance targets (first guesses; report actual numbers even if missed):
- fastText-style beats the keyword baseline on theme macro-F1 on the human set.
- Theme top-1 accuracy ≥ 75% on test, and ≥ 60% on the human set.
- "none" recall ≥ 70%.

---

## 8. Export and browser inference

### 8.1 `src/export.py`

Write `artifacts/classifier/`:
- `weights.bin`: float32 little-endian, in this order: embedding [B×d], theme W [(T+1)×d], theme b [T+1], image W [I×d], image b [I]. Flag `--int8` for per-row quantized embeddings (with a scale array).
- `meta.json`: B, d, hidden flag, FNV parameters, the tag id lists in head order, thresholds, the byte offsets of every array in `weights.bin`, and the export date.

### 8.2 `web/classifier.js`

- A plain ES module with no dependencies. `async load(baseUrl)`, then `predict(text) -> {theme, themeProb, images: [{id, prob}], allThemeProbs}`.
- Reimplement featurization exactly. FNV-1a in JS: `h ^= byte; h = Math.imul(h, 16777619) >>> 0`. Use `TextEncoder` for UTF-8 bytes.
- Mean of embedding rows, then matrix-vector products, then softmax and sigmoids, then thresholds.

### 8.3 `web/parity_test.mjs`

- Python dumps 50 test requests with their bucket ids and probabilities to `artifacts/parity.json`.
- Node checks that bucket ids match exactly and probabilities match within 1e-5 (within 1e-2 for int8). Fail loudly on any mismatch.

---

## 9. CLI (`src/predict.py`)

```
python -m src.predict "something about how life slowly runs out, use a lamp"
→ theme: mrityu (0.71)   images: deepak (0.82)
```

Flags: `--model {keyword,logreg,fasttext}`, `--top 3` (show the top 3 themes with probabilities).

---

## 10. Definition of done

- [ ] All tests pass (`pytest`, `node web/parity_test.mjs`).
- [ ] `reports/classifier_eval.md` exists, with all three models compared on test and human sets.
- [ ] `artifacts/classifier/` loads in the browser module, and the parity test passes.
- [ ] `NOTES.md` has an entry per step, including the sweep table and the final chosen config.
- [ ] Final summary to the user: chosen config, key metrics, top 3 failure patterns, and the API spend for data generation.