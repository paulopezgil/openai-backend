# AGENTS.md - Local AI Backend

## Project Overview
Django REST API wrapper for local GGUF models via llama-cpp-python.

## Architecture
```
root/
├── core/          # Django project settings (settings.py, urls.py, asgi.py)
├── ai/            # Main application (views.py, urls.py, models.py)
├── manage.py
└── pyproject.toml # Poetry-managed
```

**Intent (not yet fully implemented):**
- `core/` = project config (was described as `ai_api/`)
- `ai/` = HTTP layer (was described as `inference/`)
- Future: `ai/services/` for model_manager.py, llama_service.py

## Key Constraints
1. **Memory Management**: Llama model must be loaded ONCE at startup, never per-request
2. **Async Safety**: LLM inference is blocking—use Django async views + offload to threads
3. **Thin Views**: views.py only parses JSON, validates, calls service, returns response

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

## Development Notes
- Python 3.12+ required
- Uses poetry for dependency management (not pip/venv)
- ASGI is configured in `core/asgi.py`—ready for async views
- Routes: `/ai/` prefix (defined in `core/urls.py`)
- No database migrations needed yet (sqlite default, empty models)