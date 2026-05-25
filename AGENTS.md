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
│   ├── views.py   # API views (ChatCompletions, Embeddings, Models)
│   ├── exceptions.py  # Custom exception hierarchy + handle_exception()
│   ├── models.py  # GGUFModel (DB-backed per-model loader config)
│   ├── admin.py   # GGUFModel registered in admin
│   └── services/  # AI services
│       ├── __init__.py       # Composition root (wires RequestManager)
│       ├── request_manager.py  # Thread-safe orchestrator with VRAM lock
│       ├── model_service.py    # File paths, downloads, DB config lookup
│       └── llama_service/
│           ├── __init__.py
│           ├── llama_service.py  # Llama wrapper (load/unload/inference)
│           ├── schemas.py        # Pydantic models for llama-cpp inputs
│           └── cleaners.py       # Endpoint-specific parameter cleaning
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
- openai ^2.38.0

## Commands
```bash
# Install deps
poetry install

# Run dev server
poetry run python manage.py runserver 0.0.0.0:8000

# Check config
poetry run python manage.py check

# Shell
poetry run python manage.py shell

# Make migrations
poetry run python manage.py makemigrations ai
```

## Active Technical Context

### Current System State
- Three endpoints implemented: `POST /v1/chat/completions`, `POST /v1/embeddings`, `GET /v1/models`
- Chat completions support SSE streaming (OpenAI `text/event-stream` format)
- All endpoints are synchronous (Django `View`, not async)
- Endpoints served at `/ai/v1/chat/completions`, `/ai/v1/embeddings`, `/ai/v1/models`
- `GET /v1/completions` (legacy) is still a stub — `execute_completion` ready but no view wired
- Stream generator (`_stream_generator_wrapper`) holds VRAM lock for stream duration, releases in `finally`

### New Files (this session)
- `ai/services/__init__.py` — composition root
- `ai/services/request_manager.py` — orchestrates inference with VRAM lock
- `ai/services/llama_service/cleaners.py` — parameter cleaning per endpoint

### Model Configuration (DB-backed)
- `GGUFModel` model in `ai/models.py` stores per-model loader params: `n_ctx`, `n_gpu_layers`, `n_threads`, `seed`, `type` (chat/embedding)
- Migrations 0001 and 0002 already applied
- Registered in admin at `/admin/ai/ggufmodel/`
- `ModelService.get_loader_kwargs(model_name)` queries by filename, falls back to `{}`
- `RequestManager._prepare_model()` merges DB config with any explicit `loader_kwargs` (explicit wins)
- Auto-creation on download NOT yet implemented

### Exception System
- All custom exceptions in `ai/exceptions.py`
- `handle_exception(error)` is the single entry point for views — every view method has one `except Exception as e: return handle_exception(e)`
- See [Exception Hierarchy](#exception-hierarchy) below

## Services Layer

### Composition Root (`ai/services/__init__.py`)
```python
request_manager = RequestManager(
    engine=LlamaService(),
    model_service=ModelService(),
)
```
Views import `from ai.services import request_manager`. Never use singletons — normal classes with a single wired instance at module level.

### RequestManager
- `execute_chat(model_name, input_data, loader_kwargs=None)` → ChatCompletion | Iterator[ChatCompletionChunk]
- `execute_completion(model_name, input_data, loader_kwargs=None)` → CompletionResponse | Iterator[...]
- `execute_embedding(model_name, input_data, loader_kwargs=None)` → EmbeddingResponse
- Acquires `threading.Lock()` before model prep + inference; releases in `finally` or passes to `_stream_generator_wrapper`
- Wraps third-party exceptions in `ServiceError` before re-raising; passes `APIError` subclasses through unwrapped

### ModelService
- `list_ai_models()` → list[str] — GGUF file stems (no extension)
- `get_ai_model_path(model_name)` → str | None (returns None if not found)
- `get_loader_kwargs(model_name)` → dict — queries `GGUFModel` by filename (tries with/without `.gguf`), returns `{}` on miss
- `ai_model_exists(model_name)` → bool
- `download_ai_model(repo_id, filename)` → str — downloads via `hf_hub_download`

### LlamaService
- `load_ai_model(ai_model_path, n_ctx=2048, n_gpu_layers=0, n_threads=None, seed=-1, **kwargs)`
- `unload_ai_model()` — deletes model ref + `gc.collect()`
- Properties: `loaded_ai_model` (Optional[Llama]), `loaded_ai_model_path` (Optional[str])
- `create_chat_completion(input)`, `create_completion(input)`, `create_embedding(input)` — pass-through to llama-cpp, raise `ModelNotLoadedError` if no model loaded

### Parameter Cleaning (`cleaners.py`)
Per-endpoint functions that strip unsupported params and apply mappings:
- `clean_chat_params(body)` — strips `user`, `service_tier`, `metadata`, `reasoning_effort`, `stream_options`, `parallel_tool_calls`; maps `max_completion_tokens` → `max_tokens`; silently coerces `n` to 1
- `clean_embedding_params(body)` — strips `user`
- `clean_completion_params(body)` — stub, no current unsupported params

## Exception Hierarchy

```
APIError (500)               ← never raise directly, always use concrete subclass
├── BadRequest (400)         ← validation errors (missing model, bad params)
├── NotFound (404)
│   └── ModelNotFoundError   ← model file not on disk
├── ModelNotLoadedError (500)← inference attempted without loaded model
└── ServiceError (500)       ← wrapper for third-party/unexpected errors
```

### Exception Flow Rules
1. Every `raise` in `ai/services/` must be a concrete `APIError` subclass — never raw `APIError()`
2. Third-party exceptions caught in `RequestManager.execute_*()` are wrapped via `ServiceError(str(e)) from e` before re-raising
3. Our exceptions (`ModelNotFoundError`, `ModelNotLoadedError`, `BadRequest`) are re-raised as-is
4. `handle_exception()` in `exceptions.py` routes by `isinstance`:
   - `json.JSONDecodeError` → `{"error": "Invalid JSON body"}` (400)
   - `APIError` → `{"error": str(e)}` using `e.status_code`
   - `FileNotFoundError` → `{"error": "Model not found: ..."}` (404)
   - Any other → `{"error": "Internal server error"}` (500, logged)
5. Pydantic `ValidationError` is wrapped as `BadRequest` via a nested `try/except` in the view

## Learned Rules & Guardrails

- **Never use singletons** — use normal classes wired at the composition root (`ai/services/__init__.py`). This improves testability and removes hidden global state.
- **Never raise `APIError` directly** — always use a concrete subclass (`BadRequest`, `ModelNotFoundError`, `ModelNotLoadedError`, `ServiceError`).
- **Wrap third-party exceptions at service boundaries** — catch in `RequestManager.execute_*()` and re-raise as `ServiceError`.
- **Keep `except Exception` blocks only when they add real functionality** (lock cleanup, resource release, graceful shutdown). Remove bare re-raises that add no behavior.
- **Parameter cleaning lives in the service layer** (`cleaners.py`), not in views. Views only parse JSON, raise validations, and call services.

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
| `n` | Only single completion per request | Silently coerced to 1 |
| `stream_options` | Not implemented | Ignore |
| `user` | No user identification | Ignore |
| `service_tier` | Local-only API | Ignore |
| `metadata` | No storage | Ignore |
| `reasoning_effort` | Not for local models | Ignore |
| `parallel_tool_calls` | Not supported | Ignore |
| `max_completion_tokens` | Use `max_tokens` instead | Mapped to `max_tokens` |

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
5. **Default silently**: `n` (coerced to 1), `service_tier`, `parallel_tool_calls` (ignored)

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

## Future Roadmap & Ideas
- **Streaming timeout**: Add `_lock.acquire(timeout=30)` to prevent infinite user queuing (TODO in `request_manager.py`)
- **Async views**: Convert to Django async views + thread offloading to avoid blocking WSGI workers during inference
- **Auto-create GGUFModel on download**: When a model is downloaded via `download_ai_model()`, auto-create a `GGUFModel` row with defaults so every model has a config entry
- **Wire legacy completions**: `/v1/completions` endpoint — `execute_completion` is ready, view is a stub
- **Expose n_gpu_layers, n_ctx in API**: Allow clients to request GPU layer count or context window per-request
- **Admin download UI**: Let admins download models from HuggingFace directly through the admin interface
- **Chat history**: Add message history/context management for multi-turn conversations
- **Error sanitization**: `ServiceError` currently passes third-party error messages through — consider sanitizing for production