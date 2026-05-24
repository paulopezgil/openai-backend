# AGENTS.md - Local AI Backend

## Project Overview
Django REST API wrapper for local GGUF models via llama-cpp-python.

## Local LLM API — Mechanism Overview

### What this is

A locally hosted API that mimics the OpenAI interface so that any tool,
library, or client already built for OpenAI works out of the box — with
no code changes on the client side — but instead of sending requests to
OpenAI's servers, all inference runs on local hardware.

---

### The core problem

Running a large language model requires significant GPU memory (VRAM).
Unlike a typical web API where many requests can be handled in parallel,
only one model can realistically run at a time on consumer or prosumer
hardware. Attempting to serve multiple simultaneous requests would either
crash the process or produce degraded, unpredictable results.

---

### The mechanism

When a client sends a request, instead of processing it immediately, the
system places it in a queue and holds the connection open. A single
background worker pulls requests from that queue one at a time, runs the
model, and sends the result back to the client that was waiting. From the
client's perspective the experience is identical to calling OpenAI — it
sends a request and eventually gets a response. It has no awareness of
the queue.

```
Client A ──┐
Client B ──┼──► Queue ──► [ Worker ] ──► Model ──► Response
Client C ──┘         (one at a time)
```

Clients that arrive while another request is being processed simply wait
their turn. The slowest part of the system — the model itself — is never
asked to do more than one thing at once.

---

### Model flexibility

The system is not tied to a specific model or model format. Models are
stored locally and can be swapped by name in the request, exactly as you
would specify a model name when calling OpenAI.
If the requested model is not currently loaded in the
GPU, then the current model is unloaded and the requested
one is loaded, then called.

---

### What the client sees

To any OpenAI-compatible client the API is indistinguishable from the
real thing:

- The endpoint paths are identical (`/v1/chat/completions`, `/v1/models`)
- The request and response shapes are identical
- Authentication works the same way

The only observable difference is that responses may take longer when the
queue is busy, and an optional status endpoint reveals how many requests
are waiting.

## Architecture
```
root/
├── core/          # Django project settings (settings.py, urls.py, asgi.py)
├── ai/            # Main application
│   └── services/  # AI services
│       └── ai_model_service.py  # AIModelRegistry, AIModelLoader
├── manage.py
└── pyproject.toml # Poetry-managed
```

## Key Constraints
1. **Memory Management**: Only one model is loaded into memory/VRAM at a time. The active model is unloaded before loading a new one to prevent VRAM exhaustion.
2. **Async Safety**: LLM inference is blocking—use Django async views + offload to threads
3. **Thin Views**: views.py only parses JSON, validates, calls service, returns response

## Dependencies
- django ^5.0
- llama-cpp-python ^0.3
- huggingface-hub ^0.20.0
- pydantic ^2.13.4
- python-dotenv ^1.2.2
- django-environ ^0.13.0

## Commands
```bash
# Install deps
poetry install

# Run dev server (ASGI recommended for async)
poetry run python manage.py runserver 0.0.0.0:8000

# Check config
poetry run python manage.py check

# Shell
poetry run python manage.py shell
```

## Services Layer

### AIModelRegistry
- `download_ai_model(repo_id, filename)` - Download GGUF from HuggingFace
- `list_ai_models()` - List available GGUF files in AI models directory
- `get_ai_model_path(ai_model_name)` - Get full path to AI model file

### AIModelLoader
- `load_ai_model(ai_model_path, **kwargs)` - Load AI model into memory by path
- `clear_loaded_ai_model()` - Unload AI model from memory
- Properties: `loaded_ai_model`, `loaded_ai_model_path`

Both are normal Python classes (not singletons) implemented in `ai/services/ai_model_service.py`.

## AI Model File Handling
- AI Models stored in `./ai_models/` directory (flat files, no subfolders)
- Downloaded files saved as-is (e.g., `tinyllama-1.1b-chat-v1.0.Q2_K.gguf`)
- Use AI model name without `.gguf` extension when loading (e.g., `tinyllama-1.1b-chat-v1.0.Q2_K`)

## Chat Completion API — Parameter Compatibility

The `/v1/chat/completions` endpoint supports OpenAI-compatible parameters with the following compatibility notes for `llama-cpp-python ^0.3`:

### Fully Compatible Parameters

These parameters are fully supported with equivalent semantics:

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `messages` | list | REQUIRED | Message objects with `role` and `content` |
| `model` | str | (current) | Model name to load (optional if pre-loaded) |
| `temperature` | float | 1.0 → 0.2* | Sampling temperature (0.0–2.0) |
| `top_p` | float | 1.0 → 0.95* | Nucleus sampling threshold |
| `max_tokens` | int | None | Max output tokens (None = unlimited) |
| `stop` | str\|list | None | Stop sequences |
| `stream` | bool | False | Stream response tokens |
| `presence_penalty` | float | 0.0 | Penalty for presence in prompt (-2.0–2.0) |
| `frequency_penalty` | float | 0.0 | Penalty for frequency in prompt (-2.0–2.0) |
| `seed` | int | None | Random seed for reproducibility |
| `response_format` | object | None | `{"type": "json_object"}` for JSON mode |
| `logit_bias` | dict | None | Token ID → bias mapping |
| `logprobs` | bool | None | Include log probabilities |
| `top_logprobs` | int | None | Number of top log probabilities |
| `tools` | list | None | Function/tool definitions |
| `tool_choice` | str\|obj | None | Tool selection: `"auto"`, `"none"`, or specific |

\* *Note: llama-cpp defaults differ from OpenAI—explicitly pass values for consistent behavior.*

### Parameters NOT Supported

These OpenAI parameters have no equivalent in llama-cpp-python:

| Parameter | Reason | Handling |
|---|---|---|
| `n` | Only single completion per request | Reject if != 1 |
| `stream_options` | Not implemented | Ignore |
| `user` | No user identification | Ignore |
| `service_tier` | Local-only API | Reject or ignore |
| `metadata` | No storage | Ignore |
| `reasoning_effort` | Not for local models | Ignore |
| `parallel_tool_calls` | Not supported | Ignore or set False |
| `max_completion_tokens` | Use `max_tokens` instead | Use as fallback |

### Advanced llama-cpp Sampling Parameters (Optional)

These advanced parameters are available in llama-cpp-python for fine-tuned sampling behavior:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_k` | int | 40 | Keep top-k tokens (0 = disabled) |
| `min_p` | float | 0.05 | Minimum probability threshold |
| `typical_p` | float | 1.0 | Typical probability sampling |
| `repeat_penalty` | float | 1.0 | Penalty for repeated tokens |
| `tfs_z` | float | 1.0 | Tail-free sampling parameter |
| `mirostat_mode` | int | 0 | Mirostat algorithm (0=off, 1=v1, 2=v2.0) |
| `mirostat_tau` | float | 5.0 | Mirostat target entropy |
| `mirostat_eta` | float | 0.1 | Mirostat learning rate |

These can be optionally added to `ChatCompletionRequest` for clients requiring advanced sampling control.

### Implementation Guidelines

1. **Always pass to llama_cpp**: `messages`, `stream`, `temperature`, `top_p`, `max_tokens`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `response_format`, `tools`, `tool_choice`
2. **Pass if provided**: `logit_bias`, `logprobs`, `top_logprobs`, and advanced sampling parameters
3. **Model parameter**: Set to `None` to use currently loaded model, or pass to identify in response
4. **Ignore safely**: `user`, `metadata`, `stream_options` (accept in schema but don't forward)
5. **Reject or default**: `n` (must be 1), `service_tier`, `parallel_tool_calls` (validate or use default)

### Response Format

llama-cpp-python returns responses matching OpenAI's `ChatCompletionResponse` structure:
- `id`: unique identifier
- `object`: `"chat.completion"` or `"chat.completion.chunk"` (streaming)
- `created`: Unix timestamp
- `model`: model name used
- `choices`: list of completion choices with `message`, `finish_reason`
- `usage`: token counts (`prompt_tokens`, `completion_tokens`, `total_tokens`)

Finish reasons: `"stop"`, `"length"`, `"tool_calls"`, or `None`

See `ai/LLAMA_CPP_COMPATIBILITY.md` for detailed compatibility research.

## Future Roadmap
- Add Django async views for HTTP endpoints
- Implement streaming responses (SSE)
- Add model configuration stored in database (per-user models)
- Chat completion endpoint with message history
- Admin UI for model download/management