# openai-backend Agent Guide

## Setup & Development
- Install deps: `pip install -r api/requirements.txt`
- Run dev server: `uvicorn api.app:app --reload --host 0.0.0.0 --port 8000`
- API docs available at http://localhost:8000/docs

## Project Structure
- Main app: `api/app.py`
- API routes: `api/routes/` (chat, completions, embeddings, etc.)
- Schema definitions: `api/schemas/`
- Entry point: `api/app.py` creates FastAPI instance

## Implementation Notes
- Currently minimal implementation - most route handlers return `pass`
- Follow OpenAI API specifications for endpoint behavior
- All routes use `/v1/{resource}` prefix pattern
- CORS configured to allow all origins (for development)

## Testing
- No test suite currently implemented
- Manual testing via curl or API clients recommended