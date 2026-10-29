# Using LM Studio from Python: cheat sheet

LM Studio runs an LLM on your machine and serves it over HTTP using the **same API format as
OpenAI**. Any code that talks to OpenAI can talk to LM Studio by changing the URL.

## The `lms` CLI

| Command | What it does |
|---|---|
| `lms status` | Is the server on? Which model is loaded? |
| `lms ls` | Models downloaded to disk |
| `lms ps` | Models currently loaded in RAM (with context size, parallel slots) |
| `lms server start` / `lms server stop` | Start/stop the HTTP server (port 1234) |
| `lms load qwen/qwen3.5-9b --yes` | Load a model into memory (~16 s, ~6 GB RAM) |
| `lms unload --all` | Free the RAM when you're done |
| `lms log stream` | Watch requests arrive live (great for learning) |

## The HTTP API

```bash
# Which models can I call?
curl http://localhost:1234/v1/models

# One chat request
curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "qwen/qwen3.5-9b",
  "messages": [{"role": "system", "content": "Be brief."},
               {"role": "user",   "content": "Say hi in 3 words."}],
  "temperature": 0.9,
  "max_tokens": 100,
  "reasoning_effort": "none"
}'
```

The reply text is in `choices[0].message.content`. `usage` has token counts.

- **messages**: the conversation. `system` = standing instructions, `user` = the request.
  Sending a system message overrides any preset system prompt set in the LM Studio app.
- **temperature**: randomness. 0 = always the most likely word; ~0.9 = varied (good for data generation).
- **max_tokens**: hard cap on reply length. Always set it, so a runaway reply can't hang a script.
- **reasoning_effort: "none"**: Qwen 3.5 is a *thinking* model. Without this it writes ~1000 hidden
  reasoning tokens (`message.reasoning_content`) before answering, which is ~7x slower here.

## In this project

All LLM access goes through `src/llm_client.py` → `complete(prompt) -> str`. Settings come from env vars:

```bash
export LLM_PROVIDER=lmstudio
export LLM_MODEL=qwen/qwen3.5-9b
export LLM_BASE_URL=http://localhost:1234/v1
```

Every call is logged to `logs/llm_calls.jsonl` (prompt, reply, seconds, tokens). To look at them:

```bash
tail -n 3 logs/llm_calls.jsonl | python -m json.tool --json-lines
```
