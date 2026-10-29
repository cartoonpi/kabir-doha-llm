"""Step 2.2: generate synthetic English requests with the local LLM.

Pipeline:
  1. PLAN    -- decide every LLM call up front (theme + random images), seeded,
                so the plan is identical on every run.
  2. CALL    -- one LLM call per planned combo, asking for 5 requests as JSON.
                Raw replies are appended to a *_raw.jsonl file as they arrive,
                so an interrupted run resumes where it stopped (calls already
                in the raw file are skipped).
  3. CLEAN   -- length filter, exact dedupe, near-dedupe (Jaccard), counts.

Usage (run from the classifier/ folder):
  python -m src.gen_requests --dry-run              # show plan + sample prompt, no calls
  python -m src.gen_requests --pilot                # 1 theme, few calls, show 20 samples
  python -m src.gen_requests --combos-per-theme 60  # full run
  python -m src.gen_requests --clean-only           # re-run cleaning on existing raw file

Note on reproducibility: the PLAN and the CLEANING are deterministic. The LLM's
text is not (temperature 0.9 is random on purpose, for variety), which is why
we save the raw output and treat that file as the fixed dataset afterwards.
"""

import argparse
import json
import random
import re
import statistics
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import llm_client
from src.common import DATA, SEED, get_logger, load_tags, read_jsonl, set_seeds, write_jsonl

REQUESTS_PER_CALL = 5
NONE_FRACTION = 0.10      # ~10% of all requests match no theme
ZERO_IMAGE_FRACTION = 0.30

# Topics for "none" requests. Rotating through a list gives far more variety
# than asking the LLM "write something off-topic" 150 times.
NONE_TOPICS = [
    "cricket", "filing income tax", "a funny cat video", "fixing a laptop", "cooking biryani",
    "a job interview", "learning to drive", "the stock market", "football transfers", "a birthday party",
    "booking a train ticket", "smartphone reviews", "gym workouts", "weather tomorrow", "a math exam",
    "car insurance", "a Bollywood movie", "coding in Python", "a wedding playlist", "traffic in the city",
    "buying shoes", "monsoon rain", "a pizza order", "electricity bill", "video games",
    "space rockets", "a dog's haircut", "office meetings", "a cricket world cup final", "online shopping sales",
]

# v1 lesson: "none" requests never mentioned lamps, fish or markets, so the model
# learned "object word => some theme" ("why is the fish jumping" -> maya). These
# topics put everyday objects into plain, non-spiritual requests on purpose.
NONE_OBJECT_TOPICS = [
    "feeding the goldfish in my aquarium", "buying vegetables at the weekly market", "a boat ride on holiday",
    "planting a mango tree in the garden", "fixing a broken table lamp", "repairing a cracked clay pot",
    "a candle-light dinner reservation", "cleaning the water well pump", "a moth stuck in my room",
    "buying sandalwood soap", "a tree fell on my car", "choosing a mirror for the hallway",
    "the price of cotton this year", "a crow stealing food from the balcony", "a pearl necklace gift",
    "repairing a ceiling fan", "a cooking pot that burns food", "the moon looks huge tonight",
]
OBJECT_NONE_EVERY = 3  # every 3rd "none" call uses an object topic (~1/3 of none data)
MAX_FORBIDDEN_IMAGES = 20  # with 100 images, listing all 99 others bloats every prompt

# Writing styles from the spec. Shuffled per call so position 1 isn't always "casual".
STYLES = [
    "casual, like a quick chat message",
    "poetic and a little flowery",
    "indirect and personal: about a real-life situation (e.g. 'my grandfather's last days') without abstract theme words",
    # Don't give a concrete example here (e.g. "death + lamp"): the model copies
    # examples, which would leak one theme/image into every other theme's data.
    "very short: 2 to 5 words, keyword style, like 'topic + object'",
    "long and detailed: 25 to 40 words",
]


# --------------------------------------------------------------------------- plan

def make_plan(tags: dict, combos_per_theme: int, themes_subset: list[str] | None, rng: random.Random) -> list[dict]:
    """Return a list of planned calls: {"call_id", "theme", "images"}."""
    image_ids = [im["id"] for im in tags["images"]]
    plan = []
    for theme in tags["themes"]:
        if themes_subset and theme["id"] not in themes_subset:
            continue
        for i in range(combos_per_theme):
            r = rng.random()
            if r < ZERO_IMAGE_FRACTION:
                k = 0
            elif r < ZERO_IMAGE_FRACTION + (1 - ZERO_IMAGE_FRACTION) / 2:
                k = 1
            else:
                k = 2
            plan.append({"call_id": f"{theme['id']}-{i:03d}", "theme": theme["id"], "images": rng.sample(image_ids, k)})

    # "none" calls: sized so they make up ~10% of all requests.
    #   none / (theme + none) = 0.1  ->  none = theme / 9
    n_none = round(len(plan) * NONE_FRACTION / (1 - NONE_FRACTION))
    for i in range(n_none):
        if i % OBJECT_NONE_EVERY == OBJECT_NONE_EVERY - 1:
            topic, kind = NONE_OBJECT_TOPICS[(i // OBJECT_NONE_EVERY) % len(NONE_OBJECT_TOPICS)], "object"
        else:
            topic, kind = NONE_TOPICS[i % len(NONE_TOPICS)], "plain"
        plan.append({"call_id": f"none-{i:04d}", "theme": "none", "images": [], "topic": topic, "none_kind": kind})
    return plan


# --------------------------------------------------------------------------- prompts

def build_prompt(call: dict, tags: dict, rng: random.Random) -> str:
    themes = {t["id"]: t for t in tags["themes"]}
    images = {im["id"]: im for im in tags["images"]}
    styles = STYLES[:]
    rng.shuffle(styles)
    style_lines = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(styles))
    others = [im["name"] for iid, im in images.items() if iid not in call["images"]]
    if len(others) > MAX_FORBIDDEN_IMAGES:
        others = rng.sample(others, MAX_FORBIDDEN_IMAGES)  # seeded per call -> reproducible
    forbidden = ", ".join(others)

    intro = (
        'Users of a "Kabir doha generator" app type a short English request describing the doha '
        f"(two-line Hindi verse) they want. Write {REQUESTS_PER_CALL} different requests such a user might type.\n\n"
    )

    if call["theme"] == "none":
        # With 100 themes, list the ~10 family names instead of every theme name.
        avoid = ", ".join(f["name"] for f in tags["families"]) if "families" in tags else ", ".join(t["name"] for t in tags["themes"])
        body = (
            f"These requests must be about: {call['topic']}.\n"
            "They must NOT touch any spiritual or life-lesson theme "
            f"(avoid: {avoid}).\n"
        )
        if call.get("none_kind") == "object":
            body += "Mention the everyday object(s) in the topic plainly and literally, never as a metaphor or life lesson.\n"
        else:
            body += f"Do not mention any of these: {forbidden}.\n"
        body += ("Some can ask for a doha, poem or verse about the topic (literal, light-hearted); "
                 "others can be plain off-topic messages.\n")
    else:
        t = themes[call["theme"]]
        body = f"Every request must be about this theme: {t['name']} ({t['desc']}).\n"
        if call["images"]:
            names = " and ".join(images[i]["name"] for i in call["images"])
            body += f"Every request must mention or clearly imply: {names}.\n"
        else:
            # Pilot v1 showed "like a candle burning down" in a zero-image request:
            # that's really the lamp image, so the label would be wrong. Be explicit.
            body += "Do not use any concrete object, metaphor or comparison (no candles, lights, flames, rivers, plants, animals).\n"
        body += f"Never mention any of these: {forbidden}.\n"
        body += f'Use the exact word "{t["name"]}" in at most 1 of the {REQUESTS_PER_CALL} requests; otherwise only imply the theme.\n'

    return (
        intro + body
        + f"\nUse one style per request:\n{style_lines}\n\n"
        # Pilot v1 produced "...without mentioning death directly...": the model
        # echoed our instructions into the data.
        + "Write only what the user would type. Never refer to these instructions or rules.\n"
        + 'Return JSON only, no commentary, exactly this shape: [{"text": "..."}, {"text": "..."}]'
    )


def parse_reply(reply: str) -> list[str]:
    """Pull the list of texts out of the reply. Models sometimes wrap JSON in
    ```json fences or add a sentence, so we cut from the first '[' to the last ']'."""
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in reply")
    items = json.loads(reply[start : end + 1])
    texts = [it["text"].strip() for it in items if isinstance(it, dict) and isinstance(it.get("text"), str)]
    if not texts:
        raise ValueError("JSON array had no {'text': ...} items")
    return texts


# --------------------------------------------------------------------------- cleaning

def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|: 1.0 = same word set, 0.0 = nothing in common."""
    return len(a & b) / len(a | b) if a | b else 1.0


# Phrases that show the model echoing our prompt rules into the data
# (pilot v2: "...without saying the forbidden words").
META_LEAK = re.compile(r"forbidden|(?:these|the above|your|my) (?:instructions|rules)|without (?:mentioning|naming|using|saying) (?:the (?:word|theme|name)|it directly|\w+ directly|any (?:image|object))|the word ['\"“]|\bthe rules? (?:say|said)\b", re.I)
# v2: the v1 pattern also matched "the word ", "instruction", "the rules" -> 8/8 of its v1 drops and 75/75 of its v2
# drops were false positives ("the Word" is literally the shabd theme). The real leaks only ever came from local Qwen.

THEME_NAME_CAP = 1 / 3  # spec: theme's English name in at most ~a third of requests

# A request phrase at the START of a text: optional politeness, a verb, a poem
# word, an optional preposition. Matches "Please write a doha about", "doha pls:",
# "a doha on", "doha w a" (-> keeps "a ..."), "Can you shape a verse about",
# "I want a couplet where", "Give me a short Kabir doha on".
_POEM = r"(?:doha|dohas|verse|couplet|poem|lines|shayari|sakhi)"
FRAME = re.compile(
    r"^\s*(?:(?:please|pls|plz|kindly|hey|so|ok|okay)[,!]?\s+)*"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
    r"(?:(?:i\s+(?:want|need|would\s+like|'d\s+like)|i'd\s+like|write|compose|create|make|craft|shape|give|send|need|want)\s+(?:me\s+)?)?"
    r"(?:(?:a|an|one|some)\s+)?(?:(?:short|simple|little|small|quick|nice|good|new|kabir|kabir-style|kabir's)\s+)*" + _POEM +
    r"(?:\s+(?:pls|please|plz))?\s*(?:about|on|of|with|w|where|that|for|comparing|showing|telling)?\s*[:,\-–]?\s+",
    re.I)
FRAME_STRIP_P = 0.5

POEM_WORDS = ["doha", "poem", "verse", "couplet", "shayari"]
POEM_WEIGHTS = [50, 15, 15, 10, 10]
POEM_PLURAL = {"doha": "dohas", "poem": "poems", "verse": "verses", "couplet": "couplets", "shayari": "shayari"}
POEM_WORD = re.compile(r"\b(?:dohas?|poems?|verses?|couplets?|shayari)\b", re.I)


def clean(rows: list[dict], log, tags: dict) -> list[dict]:
    log.info("cleaning %d raw requests", len(rows))

    kept = [r for r in rows if 2 <= len(r["text"].split()) <= 40]
    log.info("  length filter (2-40 words): %d dropped -> %d", len(rows) - len(kept), len(kept))

    # Label check: a theme request that names an image NOT in its label teaches
    # the image head the wrong answer ("heart of stone" labelled without the
    # stone image). v2 writers did this in ~0.9% of texts. "none" requests are
    # exempt: none/object calls mention everyday objects on purpose.
    def phrase(name):
        parts = [re.escape(w) for w in re.sub(r"[^a-z' ]", " ", name.lower()).split() if w != "and"]
        return re.compile(r"\b" + r"\s+(?:and\s+)?".join(parts) + r"s?\b", re.I)
    image_re = {i["id"]: phrase(i["name"]) for i in tags["images"]}

    def names_unlisted_image(r):
        # Blank out the row's OWN images first: "philosopher's stone" (listed)
        # contains "stone" (another image) and must not count as a leak.
        # (v2 first pass: 202 of 245 hits were exactly this false positive.)
        text = r["text"]
        for iid in r["images"]:
            text = image_re[iid].sub(" ", text)
        return any(rx.search(text) for iid, rx in image_re.items() if iid not in r["images"])

    n = len(kept)
    kept = [r for r in kept if r["theme"] == "none" or not names_unlisted_image(r)]
    log.info("  unlisted-image filter: %d dropped -> %d", n - len(kept), len(kept))

    n = len(kept)
    for r in kept:
        if META_LEAK.search(r["text"]):
            log.debug("  meta-leak dropped: %s", r["text"])
    kept = [r for r in kept if not META_LEAK.search(r["text"])]
    log.info("  prompt-echo filter: %d dropped -> %d", n - len(kept), len(kept))

    # The LLM doesn't reliably obey "use the theme word at most once", so we
    # enforce the cap here. Per theme, keep all requests without the name, and
    # only as many with the name as keeps them at <= 1/3. Which ones survive is
    # decided by a seeded shuffle, so it's reproducible.
    names = {t["id"]: t["name"].lower() for t in tags["themes"]}
    capped, n = [], len(kept)
    for theme_id in sorted({r["theme"] for r in kept}):
        group = [r for r in kept if r["theme"] == theme_id]
        if theme_id not in names:  # "none" has no theme word
            capped += group
            continue
        pattern = re.compile(rf"\b{re.escape(names[theme_id])}\b", re.I)
        with_name = [r for r in group if pattern.search(r["text"])]
        without = [r for r in group if not pattern.search(r["text"])]
        # with / (with + without) <= cap  ->  with <= cap * without / (1 - cap)
        max_with = int(THEME_NAME_CAP * len(without) / (1 - THEME_NAME_CAP))
        random.Random(f"{SEED}-cap-{theme_id}").shuffle(with_name)
        capped += without + with_name[:max_with]
    # Restore original order so later steps (dedupe) don't depend on the grouping.
    order = {id(r): i for i, r in enumerate(kept)}
    kept = sorted(capped, key=lambda r: order[id(r)])
    log.info("  theme-name cap (<= 1/3 per theme): %d dropped -> %d", n - len(kept), len(kept))

    # Frame stripping (v2, owner-approved). Each writer had a favourite request
    # phrase ("doha pls:" 95% one family, "a doha on" 93% another), and theme
    # requests asked for a poem ~45% of the time vs 19% for "none". Both are
    # label shortcuts. For a seeded half of theme rows that START with a request
    # phrase, cut the phrase: "Please write a doha about a man who hoards" ->
    # "a man who hoards". The label still fits and users type both forms.
    n_framed = n_stripped = 0
    for r in kept:
        if r["theme"] == "none":
            continue
        m = FRAME.match(r["text"])
        if not m:
            continue
        n_framed += 1
        rest = r["text"][m.end():].strip()
        if len(rest.split()) >= 2 and random.Random(f"{SEED}-frame-{r['text']}").random() < FRAME_STRIP_P:
            log.debug("  frame stripped: %r -> %r", r["text"], rest)
            r["text"] = rest
            n_stripped += 1
    log.info("  frame stripping: %d of %d framed theme requests stripped", n_stripped, n_framed)

    # Poem-word randomization (v2, owner-approved). Theme writers almost always
    # wrote "doha"; the none writer spread "poem/verse/couplet/shayari", so in
    # training "poem" meant none 89% of the time (8.8x the base rate) and
    # "write a poem about losing someone dear, use lamp imagery" -> none (1.00).
    # Every poem word in EVERY row is redrawn from one fixed mix, so the word
    # carries no label information. "lines" is left alone (plural-only usage).
    n_words = 0

    def redraw(m, text):
        nonlocal n_words
        n_words += 1
        word = random.Random(f"{SEED}-poem-{text}-{m.start()}").choices(POEM_WORDS, weights=POEM_WEIGHTS)[0]
        if m.group(0).lower().endswith("s") and m.group(0).lower() != "shayari":
            word = POEM_PLURAL[word]
        return word.capitalize() if m.group(0)[0].isupper() else word

    for r in kept:
        original = r["text"]
        r["text"] = POEM_WORD.sub(lambda m: redraw(m, original), original)
    log.info("  poem-word randomization: %d poem words redrawn from %s", n_words, dict(zip(POEM_WORDS, POEM_WEIGHTS)))

    seen, uniq = set(), []
    for r in kept:
        key = r["text"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    log.info("  exact duplicates (lowercased): %d dropped -> %d", len(kept) - len(uniq), len(uniq))

    # Near-duplicates: compare each request to every one already kept. O(n^2),
    # but n is a few thousand, so it's a few seconds.
    # Near-duplicates are compared WITHIN each theme only. Comparing all pairs
    # is O(n^2): 25k requests = 312M comparisons (too slow in Python); per theme
    # it's 100 x 250^2 = 6M. Near-duplicates across themes are rare and harmless.
    final, kept_sets = [], {}
    for r in uniq:
        ws = set(words(r["text"]))
        group = kept_sets.setdefault(r["theme"], [])
        if any(jaccard(ws, k) > 0.8 for k in group):
            log.debug("  near-dup dropped: %s", r["text"])
            continue
        final.append(r)
        group.append(ws)
    log.info("  near-duplicates (Jaccard > 0.8): %d dropped -> %d", len(uniq) - len(final), len(final))

    theme_counts = Counter(r["theme"] for r in final)
    image_counts = Counter(i for r in final for i in r["images"])
    n_images = Counter(len(r["images"]) for r in final)
    log.info("per theme: %s", dict(sorted(theme_counts.items())))
    log.info("per image: %s", dict(sorted(image_counts.items())))
    log.info("images per request: %s", dict(sorted(n_images.items())))
    return final


# --------------------------------------------------------------------------- main

def estimate(plan: list[dict], sample_prompt: str, log) -> None:
    """The spec's 'API cost guard'. Locally there is no money cost, but there is
    a time cost, so we estimate both tokens and wall-clock time."""
    prompt_tok = len(sample_prompt) / 4          # rule of thumb: ~4 chars per token
    completion_tok = REQUESTS_PER_CALL * 30 + 20  # ~30 tokens per request + JSON overhead
    log.info("PLAN: %d calls, ~%d requests", len(plan), len(plan) * REQUESTS_PER_CALL)
    log.info("  est tokens: ~%d prompt + ~%d completion = ~%d total",
             prompt_tok * len(plan), completion_tok * len(plan), (prompt_tok + completion_tok) * len(plan))

    # Use real timings from past calls (logs/llm_calls.jsonl) when we have them.
    secs = []
    if llm_client.CALL_LOG.exists():
        secs = [c["seconds"] for c in read_jsonl(llm_client.CALL_LOG) if c["usage"].get("completion_tokens", 0) > 50]
    per_call = statistics.median(secs) if secs else 20.0
    log.info("  est time: %.1f s/call (%s) -> ~%.0f min with 1 worker",
             per_call, f"median of {len(secs)} logged calls" if secs else "guess", per_call * len(plan) / 60)
    log.info("  cost: $0 (local LM Studio, model %s)", llm_client.MODEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and a sample prompt; make no calls")
    ap.add_argument("--pilot", action="store_true", help="one theme, a few calls, show 20 samples")
    ap.add_argument("--pilot-theme", default="mrityu")
    ap.add_argument("--combos-per-theme", type=int, default=150)
    ap.add_argument("--workers", type=int, default=1, help="parallel requests to LM Studio")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--clean-only", action="store_true", help="skip generation, just clean the raw file")
    ap.add_argument("--export-plan", action="store_true",
                    help="write data/plan.jsonl (every call + its prompt) for an external writer, e.g. a Claude subagent")
    args = ap.parse_args()

    log = get_logger("gen_requests_pilot" if args.pilot else "gen_requests")
    set_seeds()
    tags = load_tags()
    rng = random.Random(SEED)

    if args.pilot:
        plan = make_plan(tags, combos_per_theme=6, themes_subset=[args.pilot_theme], rng=rng)
        plan = [c for c in plan if c["theme"] != "none"]  # pilot = one theme only
        raw_path, out_path = DATA / "requests_pilot_raw.jsonl", DATA / "requests_pilot.jsonl"
    else:
        plan = make_plan(tags, args.combos_per_theme, None, rng)
        raw_path, out_path = DATA / "requests_raw.jsonl", DATA / "requests_synth.jsonl"

    # Prompts are built from a separate seeded RNG per call, so a call's prompt
    # is the same whether or not earlier calls were skipped on resume.
    prompts = {c["call_id"]: build_prompt(c, tags, random.Random(f"{SEED}-{c['call_id']}")) for c in plan}

    if args.export_plan:
        # Alternative to calling LM Studio: hand the exact same prompts to another
        # writer. It must append {call_id, theme, images, texts} lines to the raw
        # file; then `--clean-only` runs the same cleaning as the LLM path.
        plan_path = DATA / "plan.jsonl"
        write_jsonl(plan_path, [{**c, "prompt": prompts[c["call_id"]]} for c in plan])
        log.info("wrote %d planned calls to %s", len(plan), plan_path.relative_to(DATA.parent))
        return

    if not args.clean_only:
        done = {r["call_id"] for r in read_jsonl(raw_path)} if raw_path.exists() else set()
        todo = [c for c in plan if c["call_id"] not in done]
        log.info("%d calls planned, %d already in %s, %d to do", len(plan), len(done), raw_path.name, len(todo))
        estimate(todo, prompts[plan[0]["call_id"]], log)

        if args.dry_run:
            log.info("--- sample prompt (%s) ---\n%s", plan[0]["call_id"], prompts[plan[0]["call_id"]])
            none_calls = [c for c in plan if c["theme"] == "none"]
            if none_calls:
                log.info("--- sample 'none' prompt ---\n%s", prompts[none_calls[0]["call_id"]])
            log.info("dry run: no calls made")
            return
        if todo and not args.yes:
            if input("Proceed with these calls? [y/N] ").strip().lower() != "y":
                log.info("aborted by user")
                return

        lock = threading.Lock()  # one writer at a time when workers > 1
        failures = 0

        def run(call):
            # Up to 3 attempts: small models occasionally return broken JSON.
            for attempt in range(3):
                reply = llm_client.complete(prompts[call["call_id"]])
                try:
                    return call, parse_reply(reply)
                except (ValueError, json.JSONDecodeError) as e:
                    log.warning("%s attempt %d: bad reply (%s): %r", call["call_id"], attempt + 1, e, reply[:200])
            return call, None

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, c) for c in todo]
            for n, fut in enumerate(as_completed(futures), 1):
                call, texts = fut.result()
                if texts is None:
                    failures += 1
                    continue
                with lock, open(raw_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({**call, "texts": texts}, ensure_ascii=False) + "\n")
                t = llm_client.usage_totals
                log.info("[%d/%d] %s images=%s -> %d texts | avg %.1fs/call, %d tokens so far",
                         n, len(todo), call["call_id"], call["images"], len(texts),
                         t["seconds"] / max(t["calls"], 1), t["prompt_tokens"] + t["completion_tokens"])
                for tx in texts:
                    log.debug("    %s", tx)

        log.info("generation done: %d failed calls; usage this run: %s", failures, llm_client.usage_totals)

    # Flatten raw calls into one row per request, in the spec's schema.
    raw = read_jsonl(raw_path)
    rows = [{"text": tx, "theme": c["theme"], "images": c["images"], "source": "synth"} for c in raw for tx in c["texts"]]
    final = clean(rows, log, tags)
    write_jsonl(out_path, final)
    log.info("wrote %d requests to %s", len(final), out_path.relative_to(DATA.parent))

    if args.pilot:
        log.info("--- 20 random pilot samples ---")
        for r in random.Random(SEED).sample(final, min(20, len(final))):
            log.info("  [%s %s] %s", r["theme"], ",".join(r["images"]) or "-", r["text"])


if __name__ == "__main__":
    main()
