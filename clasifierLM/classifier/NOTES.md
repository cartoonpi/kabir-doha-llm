# NOTES

One entry per step: what was built, key numbers, open issues.

## Step 0 — Environment (2026-10-16)

- Python 3.14.7, venv at `.venv/`. Installed: torch 2.14.1+cpu (CPU wheel; no usable GPU),
  numpy 2.5.4, scikit-learn 1.9.1, pyyaml 6.0.3, pytest 9.1.1. Exact versions in `requirements.lock.txt`.
- Node v26.8.2 (for the step 8 parity test).
- Machine: 12 CPU threads, 7 GB RAM, no GPU. The LLM is CPU-only.
- LLM provider: **LM Studio** (local, OpenAI-compatible API at `http://localhost:1234/v1`),
  model `qwen/qwen3.5-9b`. See `LMSTUDIO.md`.

## Step 2.2 — Request generation: pilot (2026-10-16)

Built `src/common.py` (paths, logging, seeds), `src/llm_client.py`, `src/gen_requests.py`.

Key findings:
- **Qwen 3.5 thinks by default**: 1024 hidden reasoning tokens, 155 s for one call.
  `"reasoning_effort": "none"` in the request body turns it off (~22 s/call). `/no_think` and
  `chat_template_kwargs.enable_thinking=false` did *not* work in this LM Studio version.
- Speed: ~22 s per 5-request call, ~300 prompt + ~140 completion tokens per call.
- Pilot (theme `mrityu`, 6 calls): 0 JSON failures.
  - v1: zero-image requests used image-like metaphors ("like a candle burning down" = lamp → wrong
    label), and one request echoed the prompt ("without mentioning death directly"). Prompt tightened.
  - v2: images fixed; theme word "death" in 43% (> 1/3 target); one echo ("forbidden words").
  - Added two cleaning rules beyond the spec: **prompt-echo filter** and **theme-name cap (≤1/3 per theme)**.
    v2 after cleaning: 30 → 23 kept (77%), "death" in 30%.

Open issues:
- Spec plan (150 combos/theme) = 1,667 calls ≈ 10 h on this machine. Awaiting owner's choice of size.
- Labels for zero-image requests are only as good as the LLM's obedience; spot-check after the full run.

## Step 2.2 — Decision: data writer switched to a Claude Sonnet subagent (2026-10-16)

- Owner chose to have a Claude Sonnet subagent write the dataset instead of the local LLM (10 h → much faster).
- Same pipeline, only the "call" stage replaced:
  `python -m src.gen_requests --combos-per-theme 50 --export-plan` → `data/plan.jsonl` (556 calls, every prompt included)
  → subagent writes `data/raw_parts/<theme>.jsonl` (same raw format) → concatenated into `data/requests_raw.jsonl`
  → `python -m src.gen_requests --clean-only` (same cleaning as the LLM path).
- Plan: 10 themes × 50 calls + 56 "none" calls; images per call: 0→183, 1→200, 2→173. 2,780 raw requests.
- `src/llm_client.py` (LM Studio) stays in place for any future LLM steps.
- Caveat: one model writing everything = less stylistic variety than mixing writers; watch the human-set gap in step 7.
- Result: 556/556 calls, 2,780 raw texts, labels match plan exactly; independent check found 0 image words in zero-image texts.
- Cleaning: 2,780 → 2,738 (8 prompt-echo, 0 theme-cap, 23 exact dup, 11 near-dup). Themes 238–250 each, none 279;
  images 243–301 each; images/request 0:900, 1:982, 2:856.
- Open issue: "write/compose a doha…" opens 11–20% of theme requests but 0% of "none" requests → a spurious
  "not none" cue. Real users will type "write a doha about cricket". Proposed fix: add ~50 doha-phrased "none" requests.
- Subagent cost: ~317k tokens, 32 min.
- Fix applied: subagent wrote 10 extra "none" calls (`raw_parts/none_doha.jsonl`, 50 doha-phrased everyday requests).
  "write/compose a doha" opener share in none: 0% → 6% (themes 11–20%). Weakened, not eliminated; watch none-recall on human set.
- Final: 2,788 requests (none 329, themes 238–250).

## Step 2.3 — Splits (2026-10-16)

- `src/split_data.py`: stratified 80/10/10 per theme, seeded per theme. train 2,230 / val 279 / test 279.
- 0 texts shared between splits; re-running gives byte-identical files (md5 checked).
- Known caveat: the 5 texts from one LLM call share a scene/images and can land in different splits →
  test scores are a bit optimistic. The human set is the honest check.
- `data/requests_human.jsonl` created with 3 example lines. **Owner to write ~30.**

## Step 3 — Featurization (2026-10-16)

- `src/featurize.py`: words → "<w>" + char 3-grams → 32-bit FNV-1a (UTF-8) → % B (16384). `python -m src.featurize "text"` shows each step.
- `tests/test_featurize.py`: 9 tests pass (published FNV vectors "", "a", "foobar" + an independent reimplementation;
  pieces for cat/a/ok; punctuation; non-ASCII; empty → []). `pytest.ini` puts repo root on sys.path.
- Train set: 10–224 pieces per request (median 81); 5,498 of 16,384 buckets used.
- Non-ASCII (Devanagari, accents) is dropped entirely — fine for English requests, but Hinglish-in-Devanagari would featurize to nothing.
- Open: empty feature list must make the model predict "none" (handle in step 5).
- Human set: owner wrote it; file had mixed formats (JSONL, trailing commas, pretty-printed, duplicates).
  Cleaned to 56 unique JSONL rows, `[""]` → `[]`, labels untouched. Backup in `logs/requests_human_backup_*.jsonl`.
  Per theme: none 13, maya 6, mrityu 6, guru 5, vani 5, dhairya 4, pakhand 4, prem 4, ahankar 3, bhakti 3, satsang 3.
  Several image labels don't appear in the text (owner reviewing).

## Step 4 — Baselines (2026-10-16)

- `config/keywords.yaml` (6–9 keywords per tag), `src/baselines.py`: KeywordModel, LogRegModel (CountVectorizer + LogisticRegression,
  OneVsRest for images). Shared interface `predict(texts) -> [{theme_probs, image_probs}]` for evaluate.py.

| model | set | theme acc | theme macro-F1 | image micro-F1 |
|---|---|---|---|---|
| keyword | val | 0.523 | 0.535 | 0.975 |
| keyword | human | 0.554 | 0.515 | 0.696 |
| logreg | val | 0.835 | 0.835 | 0.982 |
| logreg | human | 0.661 | 0.602 | 0.667 |

- Image F1 ~0.98 on val: synthetic images are always named explicitly → images are an easy task on synthetic data.
- Val→human gap for logreg theme: 0.835 → 0.661 (synthetic-vs-real distribution shift).
- Spurious cue: logreg top "none" words include "verse", "couplet" — the 50 extra none_doha texts overused them. Over-correction.
- Human-set labels: owner approved 17 fixes (image labels not present in text removed; "potter…could not shape my ego"
  guru→ahankar; "send a text message to my friend" vani→none; none-with-image → []). Backup in logs/.
  Baselines on corrected human set: keyword theme acc 0.571 / macro-F1 0.549 / image micro-F1 0.846; logreg 0.696 / 0.636 / 0.857.

## Step 5 — fastText-style model (2026-10-17)

- `config/train.yaml`, `src/model.py` (FastTextClassifier + FastTextPredictor), `src/train.py` (single run, `--adam`, `--sweep`).
- **Bug 1, SGD didn't learn** (val acc stuck at 0.118 = always "none"). Adam learned fine (0.853), so it was step size:
  PyTorch losses average over the batch (grads ÷32), fastText updates per example. Fix: backprop `loss * batch_size`.
- **Bug 2, hidden=True never learned** (all 12 sweep runs at 0.019). Experiments (scratchpad scripts):
  random heads: no; zero hidden bias alone: no; measured 56–100% dead ReLUs. Cause: h ≈ 0.002 vs hidden bias ≈ ±0.18.
  Fix (hidden mode only, deviation from spec): embeddings uniform ±1, hidden bias 0. Hidden now 0.70–0.78 at lr 0.1;
  still dies at lr 0.5. Default (no hidden) init is exactly the spec's.
- Sweep (24 runs, 21 s total), top rows (full table in `reports/sweep.md`):

| d | B | hidden | lr | params | best epoch | val theme acc | val theme macro-F1 | val image micro-F1 | sec |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 16384 | False | 0.5 | 1,049,941 | 12 | 0.835 | 0.832 | 0.958 | 0.9 |
| 32 | 16384 | False | 0.5 | 524,981 | 14 | 0.832 | 0.830 | 0.964 | 1.2 |
| 16 | 16384 | False | 0.5 | 262,501 | 12 | 0.821 | 0.818 | 0.939 | 0.9 |
| 64 | 4096 | False | 0.5 | 263,509 | 17 | 0.821 | 0.817 | 0.966 | 1.0 |
| 32 | 4096 | False | 0.5 | 131,765 | 10 | 0.817 | 0.814 | 0.958 | 0.7 |
| 16 | 4096 | False | 0.5 | 65,893 | 14 | 0.810 | 0.808 | 0.938 | 0.9 |
| 64 | 16384 | True | 0.1 | 1,054,101 | 11 | 0.778 | 0.782 | 0.866 | 0.9 |

- **Chosen: d=32, B=16384, hidden off, SGD lr 0.5** (val theme acc 0.832, macro-F1 0.830, image micro-F1 0.964, best epoch 14).
  d=64 scored 0.832 (+0.002 ≈ <1 val example = noise) at 2× the size; picked the smaller one. Owner can override.
- Reproducible: two runs → identical checkpoint md5. 524,981 parameters.
- Spot check failures: "my boss thinks he knows everything" → prem 0.96 ("knows" is in 15/33 prem train rows: "the heart knows").
  "who won the cricket match" → none ✓ but bazaar image ≥ 0.5 (threshold to be calibrated in step 6).

## Step 6 — Calibration (2026-10-19)

- `src/calibrate.py`: `decide()` = the one thresholding rule (used by evaluate/predict; JS mirrors it). Calibrated on val for fasttext and logreg.
- fasttext: t_theme = 0.20 (grid minimum; any higher only lowers val macro-F1, since synthetic "none" requests are easy),
  t_img = 0.60 (image micro-F1 0.966 → 0.973). logreg: t_theme 0.20, t_img 0.30.
- Caveat: t_theme would matter most on vague real requests, which only exist in the human set (can't tune there).

## Step 7 — Evaluation (2026-10-19)

- `src/evaluate.py` → `reports/classifier_eval.md`; hand-written summary in `reports/classifier_eval_summary.md` (included verbatim).
- Official `fasttext` reference skipped: C++ package doesn't build on Python 3.14.

| model | set | top-1 | top-2 | macro-F1 | none P | none R | img micro-F1 | img exact |
|---|---|---|---|---|---|---|---|---|
| keyword | test | 0.534 | 0.649 | 0.548 | 0.24 | 0.73 | 0.978 | 0.961 |
| keyword | human | 0.571 | 0.643 | 0.549 | 0.45 | 0.93 | 0.846 | 0.857 |
| logreg | test | 0.824 | 0.921 | 0.824 | 0.73 | 0.97 | 0.985 | 0.971 |
| logreg | human | 0.714 | 0.786 | 0.643 | 0.88 | 1.00 | 0.857 | 0.875 |
| fasttext | test | 0.817 | 0.914 | 0.814 | 0.89 | 0.97 | 0.972 | 0.946 |
| fasttext | human | 0.679 | 0.786 | 0.629 | 0.85 | 0.79 | 0.800 | 0.821 |

- All 5 acceptance targets met. fasttext ≈ logreg (within noise on n=56). ~15-point synthetic→human drop.
- Failure patterns: "death magnet" for melancholy words; image words → not-none; overconfident on short unseen text ("make a panini" → satsang 0.72).

## Step 8 — Export + browser inference (2026-10-19)

- `src/export.py` → `artifacts/classifier/weights.bin` (2.10 MB, float32 LE, 4-byte-aligned arrays) + `meta.json`;
  `artifacts/parity.json` = 50 test requests + 11 edge cases (empty, punctuation, emoji, Devanagari, "İstanbul", 300-char word).
- `web/classifier.js`: dependency-free ES module, `load(baseUrl)` / `create(meta, buffer)` / `predict(text)`.
- `web/parity_test.mjs`: **PARITY OK** — all bucket ids identical, max prob diff 4.4e-7, decisions identical.
  - First run found image *order* differences when two images are ~1.0 (float32 ties in PyTorch, not in JS float64).
    Fix: `decide()` returns images in tag-file order (Python + JS). Metrics unchanged (set-based).
  - `--int8` (0.59 MB) misses the 1e-2 tolerance on "lamp lamp lamp" (3.2e-2; short inputs average few rows, so
    quantization error doesn't cancel) + 2 decision flips. Not shipped; export dir removed. Flag kept; the test checks it if exported.
- `tests/test_export.py`: weights.bin round-trips the checkpoint byte-exactly; meta matches featurizer. `pytest`: 11 passed.

## Step 9 — CLI + test page (2026-10-19)

- `python -m src.predict [--model keyword|logreg|fasttext] [--top N] "text"`.
- `web/index.html`: live test page (predicts as you type, probability bars with threshold lines, piece debug).
  Run: `python -m http.server 8000` in classifier/, open http://localhost:8000/web/. Verified over HTTP from Node: same output as CLI.
- Observed: "who won the cricket match" → none ✓ but bazaar image 1.00 even at t_img 0.6 (image head false alarm on none requests).

## Definition of done

- [x] All tests pass: `pytest` (11), `node web/parity_test.mjs` (PARITY OK)
- [x] `reports/classifier_eval.md`: keyword / logreg / fasttext on test + human
- [x] `artifacts/classifier/` loads in the browser module; parity passes
- [x] NOTES entry per step, sweep table, chosen config (d=32, B=16384, no hidden, SGD lr 0.5, t_theme 0.20, t_img 0.60)
- Data-generation spend: LM Studio pilot ≈ 14 local calls ($0); Sonnet subagent ≈ 320k tokens (2,830 raw requests).

---

# v2 — 10× tags (100 themes in 11 families + none, 100 images) (2026-10-20)

Plan: `~/.claude/plans/if-i-have-to-majestic-cascade.md`. Size math: params = B·d + (d+1)(T+1) + (d+1)·I.
Tags only touch the heads (693 → 6,633 params); size is set by B·d. Expected pick d=64, B=32768 ≈ 2.1M params ≈ 8.4 MB float32.
Measured on v1 data: 2,788 requests → 7,019 distinct pieces (+~1,300 per doubling) → estimate 13–15k pieces for v2.

- **Backups:** `config/tags_v1.yaml`, `config/keywords_v1.yaml`, `data/v1/`, `artifacts/v1/`, `reports/classifier_eval_v1.md`.
- **Vocabulary:** Sonnet subagent drafted `config/tags_draft.yaml` (100 themes with `family`, 11 families, 100 images; v1 20 kept verbatim).
  Owner-approved rev 2: merged kshanik+swapna, ahankar+mainpan, lobh+trishna, sampraday+ekeshwar; added yash, shunya, kirat, shoor;
  image pinjar→kankal. Promoted to `config/tags.yaml`. Near-synonym review list kept in the file header.
- **Code changes (regression-tested on v1: identical cleaned data md5, identical metrics, 11 tests pass, parity OK):**
  - gen_requests: 1/3 of none calls use everyday-object topics (`none_kind: object`); none prompt lists families not 100 themes;
    "never mention" capped at 20 random other images per call; near-dup check per theme (O(n²) → per-theme).
  - train: sweep over any keys in train.yaml; v2 grid d {32,64,100} × B {16k,32k,65k} × lr {0.1,0.5} × pos_weight_cap {10,30} = 36 runs.
  - evaluate: top-3/top-5; >12 themes → top-20 confused pairs + family confusion matrix + family accuracy; per-theme F1 worst-first.
  - test_export: B checked against checkpoint, not the featurize default. Web page shows top-10 bars.
- **Data plan:** 5,556 calls (5,000 theme + 556 none: 371 plain / 185 object) → 27,780 raw texts.
  12 parallel Sonnet subagents (one per family + none), shared instructions in scratchpad `writer_instructions.md`.
- **keywords.yaml v2** (subagent): 100 themes (5–8 kw) + 100 images (3–6), v1 entries unchanged, no keyword shared within themes/images.
  Bug found on load: bare `true` in satya's list parsed as a YAML boolean → quoted. Known overlap: "bad company" (kusang) also
  hits "company" (satsang) and the tie goes to satsang (earlier in tags.yaml).
- **Rate limit (2026-10-20):** all 12 writers stopped on the session usage limit at 1,865/5,556 calls (34%, 0 broken lines).
  Resumed all 12 at 12:11 IST, each continuing from its saved call_ids.
- **Independent check of partial output (11,965 texts):** my checker v1 split image names into words ("blind man" → "man")
  → 973 false positives; fixed to whole-phrase matching → 102 real leaks (0.85%, mostly "heart of stone").
  Added cleaning rule **unlisted-image filter** (theme rows naming an image outside their label are dropped; none rows exempt).
  On v1 data it drops 0 rows (v1 unchanged). Opener check: only fitrat at 16% (> 15% target, harmless).
- Second usage limit at 5,297/5,556 calls; 5 writers resumed with exact remaining work (259 calls). All 12 done.
  Data spend v2 ≈ 2.55M Sonnet subagent tokens (12 writers) + ~0.14M (vocabulary + keywords).
- **Full independent check:** 101/101 theme files, 27,780 texts, all call_ids/labels match plan, no theme > 15% "write/compose a doha".
- **Writer fingerprints found:** "doha pls" 95% samaj, "a doha on" 93% karmkand, "doha w a" 92% yatna, "write a verse" 77% swayam;
  poem-word share theme 39–60% vs none 19%. → owner-approved **frame stripping** (seeded 50% of framed theme rows lose the
  request phrase; `FRAME` regex unit-tested on 12 cases). After: poem-word share themes 28% vs none 19%; fingerprint openers halved (60–85 rows each).
- **Cleaning fixes:**
  - unlisted-image filter: 202 of 245 hits were "philosopher's stone" containing "stone" → blank the row's own images first. 46 real leaks dropped.
  - prompt-echo filter: old regex ("the word ", "instruction", "the rules") dropped 75/75 good v2 rows (22 were shabd = "the Word")
    and 8/8 good v1 rows. Narrowed; still catches both Qwen pilot leaks. (v1 data would now keep those 8 rows.)
- **v2 dataset:** 27,780 → 27,657 (46 image leaks, 56 exact dups, 21 near-dups; 4,247 frames stripped). Themes 227–250, none 2,770.
  Splits: train 22,119 / val 2,769 / test 2,769; 101 classes in each; ~25 per theme in val/test.
- Distinct train pieces: 14,582 (plan estimate 13–15k ✓). Bucket sharing: 33.3% @16k, 18.5% @32k, 9.9% @65k (as predicted).

## v2 Step 4–5 — Baselines, training (2026-10-20)

- Baselines on val (101 classes): keyword acc 0.328 / macro-F1 0.306 / image 0.878; logreg 0.792 / 0.785 / 0.981.
  Human set (56 rows, only v1's 11 themes): not representative of 101 classes any more — needs new human rows.
- fastText default (d=32, B=16k): theme macro-F1 0.724 in 13 s, but **image micro-F1 0.30** (v1: 0.96).
  Experiment (λ = image loss weight, cap = pos_weight cap), d=32:
  λ1/cap10 theme 0.724 img 0.283 · λ1/cap30 0.711/0.384 · λ5/cap10 0.689/0.679 · λ5/cap30 0.683/0.736.
  → a trade-off: 32 numbers in h are shared by 101 themes + 100 images. Bigger d should relieve it.
- Added `select_metric` (train.yaml): `theme` = spec/v1; `mean` = (theme macro-F1 + image micro-F1)/2, used for early stopping,
  best checkpoint and sweep winner in v2. Sweep: d {32,64,100} × B {16k,32k,65k} × cap {10,30} × λ {1,5} = 36 runs, lr 0.5.
- **Sweep result (36 runs, `reports/sweep.md`):** d 32→64 big gain, 64→100 small; B 16k→65k no gain at 4× size. Top pick (d100 B65k λ5 cap30)
  tied d100 B16k to 3 decimals. Best runs peaked at epoch 23–25 of 25 → undertrained.
  Follow-up (logs/bigger_experiment_*.log): d100 B16k 50 epochs → theme 0.765 / image 0.894; d150 identical to 4 decimals (verified real:
  2.49M vs 1.66M params); d200 0.768 / 0.898 at 13.3 MB. **Owner chose d=100, B=16384, 50 epochs, λ5, cap30 → 1,658,701 params, 6.63 MB.**
- First v2 eval: test top-1 0.783 / top-3 0.900. Found shortcut **"poem" → none 89%** (my none-writer instruction spread poem words; theme
  writers wrote "doha"). Owner-approved **poem-word randomization** (every row; mix doha 50/poem 15/verse 15/couplet 10/shayari 10; 7,314 words
  redrawn) → each poem word → none 5.4–6.0%. "lines" left alone (122 rows, still 70% none).

## v2 final (2026-10-21)

- Data 27,661; splits 22,129 / 2,766 / 2,766. Train: best epoch 32, val theme macro-F1 0.758, image 0.914 (score 0.822).
- Calibration: fasttext t_theme 0.35, t_img 0.50 (v1: 0.20 / 0.60). logreg 0.20 / 0.30.
- Test: fasttext top-1 0.783, top-3 0.899, top-5 0.927, macro-F1 0.780, none P/R 0.71/0.90, image micro-F1 0.913; family acc 0.818.
  logreg 0.780 / 0.911 / 0.936 / 0.795, image 0.980. keyword 0.322 / 0.305.
- Human (56 v1-labelled rows): fasttext top-1 0.464 — **not meaningful until relabelled to v2** (owner doing it + adding more).
- Export 6.63 MB float32; parity OK (max diff ~1e-5 — at d=100 float32-vs-float64 accumulation gets close to the 1e-5 tolerance); pytest 11 passed.
- Open: object-none topics make "lamp" a mild none cue (2.2×); top confusions are X → none (t_theme 0.35); "losing someone dear" → durjan 0.38.

## Human-set checker (2026-10-23)
Added `src/check_human.py`. It validates `data/requests_human.jsonl` against `config/tags.yaml`
(parse, empty text, unknown theme/image ids, duplicates) and lists themes with fewer than `--min` requests.
`--fix` rewrites the file as clean JSONL. Status today: 56 rows, 0 errors, but only 11/101 themes and
10/100 images covered (still the v1 labels), so 90 themes have no human requests yet.
