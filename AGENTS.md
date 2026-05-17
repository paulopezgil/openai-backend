# AGENTS.md - Local AI Backend

## Project Overview
Django REST API wrapper for local GGUF models via llama-cpp-python.

## Architecture
```
root/
├── core/          # Django project settings (settings.py, urls.py, asgi.py)
├── ai/            # Main application
│   └── services/  # AI services
│       ├── model_registry.py  # Download, list, path resolution
│       └── model_loader.py    # Load/unload LLMs, generate completions
├── manage.py
└── pyproject.toml # Poetry-managed
```

## Key Constraints
1. **Memory Management**: Llama model must be loaded ONCE at startup, never per-request
2. **Async Safety**: LLM inference is blocking—use Django async views + offload to threads
3. **Thin Views**: views.py only parses JSON, validates, calls service, returns response

## Dependencies
- django ^5.0
- llama-cpp-python ^0.2.90
- huggingface-hub ^0.20.0

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

### ModelRegistry
- `download_model(repo_id, filename)` - Download GGUF from HuggingFace
- `list_models()` - List available GGUF files in models directory
- `create_model_path(model_name)` - Get full path to model file

### ModelLoader
- `load_model(model_name, **kwargs)` - Load model into memory (thread-safe)
- `clear_loaded_model()` - Unload model from memory
- `complete(prompt, model, **kwargs)` - Generate text completion
- Properties: `loaded_model`, `loaded_model_name`, `is_loaded`

Both are singletons (one instance globally).

## Model File Handling
- Models stored in `./models/` directory (flat files, no subfolders)
- Downloaded files saved as-is (e.g., `tinyllama-1.1b-chat-v1.0.Q2_K.gguf`)
- Use full filename including `.gguf` extension when loading

## Future Roadmap
- Add Django async views for HTTP endpoints
- Implement streaming responses (SSE)
- Add model configuration stored in database (per-user models)
- Chat completion endpoint with message history
- Admin UI for model download/management