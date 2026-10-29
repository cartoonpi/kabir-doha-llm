"""Thin wrapper around the LLM. The rest of the code only ever calls `complete()`.

Provider: LM Studio running locally. LM Studio exposes an *OpenAI-compatible*
HTTP API, so a request is just a JSON POST to
    http://localhost:1234/v1/chat/completions
with {"model": ..., "messages": [...]} and the reply text is in
    response["choices"][0]["message"]["content"].

We use Python's built-in urllib instead of the `openai` or `requests` packages,
so there is no extra dependency, and you can see the raw HTTP request.

Configuration (environment variables, all optional):
    LLM_PROVIDER   "lmstudio" (the only one implemented so far)
    LLM_MODEL      default "qwen/qwen3.5-9b"   (see `lms ls` / `lms ps`)
    LLM_BASE_URL   default "http://localhost:1234/v1"
    LLM_API_KEY    not needed for LM Studio; sent as a Bearer token if set

Every call is appended to logs/llm_calls.jsonl (prompt, reply, tokens,
seconds), so you can read exactly what the model was asked and what it said.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime

from src.common import LOGS

PROVIDER = os.environ.get("LLM_PROVIDER", "lmstudio")
MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.5-9b")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")

CALL_LOG = LOGS / "llm_calls.jsonl"

# Running totals for this process, so scripts can report the total "spend".
usage_totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def complete(
    prompt: str,
    system: str = "You generate training data. Follow the format exactly.",
    temperature: float = 0.9,
    max_tokens: int = 600,
    timeout: float = 300.0,
) -> str:
    """Send one prompt, return the model's reply text."""
    if PROVIDER != "lmstudio":
        raise NotImplementedError(f"LLM_PROVIDER={PROVIDER!r} not implemented yet")

    body = {
        "model": MODEL,
        "messages": [
            # Sending our own system message replaces any preset system prompt
            # configured in the LM Studio UI, so results don't depend on it.
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        # Higher temperature = more varied wording, which we want for training data.
        "temperature": temperature,
        # Hard cap on output length, so a runaway reply can't take minutes.
        "max_tokens": max_tokens,
        # Qwen 3.5 is a "thinking" model: by default it writes ~1000 hidden
        # reasoning tokens before answering (2.5 min/call on this CPU).
        # LM Studio's reasoning_effort="none" turns that off (~3-20 s/call).
        # Note: "/no_think" in the prompt and chat_template_kwargs did NOT work.
        "reasoning_effort": "none",
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach LM Studio at {BASE_URL} ({e}). "
            "Is the server running? Try: lms server start && lms load " + MODEL
        ) from e
    seconds = time.time() - start

    text = data["choices"][0]["message"]["content"] or ""
    usage = data.get("usage", {})

    usage_totals["calls"] += 1
    usage_totals["prompt_tokens"] += usage.get("prompt_tokens", 0)
    usage_totals["completion_tokens"] += usage.get("completion_tokens", 0)
    usage_totals["seconds"] += seconds

    LOGS.mkdir(exist_ok=True)
    with open(CALL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "seconds": round(seconds, 2),
            "usage": usage,
            "finish_reason": data["choices"][0].get("finish_reason"),
            "prompt": prompt,
            "reply": text,
        }, ensure_ascii=False) + "\n")

    return text


if __name__ == "__main__":
    # Smoke test: python -m src.llm_client "Say hi in 3 words."
    import sys

    p = sys.argv[1] if len(sys.argv) > 1 else "Say hi in 3 words."
    print(complete(p, temperature=0.2, max_tokens=50))
    print(usage_totals)
