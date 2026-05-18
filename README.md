# Local AI Backend

Django REST API wrapper for local GGUF models via llama-cpp-python. OpenAI-compatible endpoints for chat completions and embeddings.

## Features

- **OpenAI-compatible API**: `/v1/chat/completions` and `/v1/embeddings` endpoints
- **Local GGUF Models**: Run quantized models without cloud dependencies
- **Model Management**: Download and manage models from HuggingFace Hub
- **Memory-efficient**: Model loaded once at startup, shared across requests
- **Async-ready**: Django async views for non-blocking inference

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | Django 5.0+ |
| AI Inference | llama-cpp-python ^0.2.90 |
| Model Downloads | huggingface-hub |
| Validation | Pydantic 2 |
| Environment | django-environ |

## Getting Started

### Prerequisites

- Python 3.12+
- Poetry

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd local-ai-backend

# Install dependencies
poetry install

# Apply migrations
poetry run python manage.py migrate

# Run the development server
poetry run python manage.py runserver 0.0.0.0:8000
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# HuggingFace token for downloading models (optional but recommended)
HF_TOKEN=hf_your_token_here

# Models directory (optional, defaults to ./models)
MODELS_DIR=./models
```

Get your HF token at: https://huggingface.co/settings/tokens

### Downloading Models

```bash
# Download a model from HuggingFace
poetry run python tests/models/download.py TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q2_K.gguf

# List available models
poetry run python tests/models/list_models.py
```

## API Endpoints

### Chat Completions

```bash
POST /v1/chat/completions
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Embeddings

```bash
POST /v1/embeddings
```

```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model.gguf",
    "input": "Your text here"
  }'
```

### Model Management

```bash
GET /v1/models          # List all available models
GET /v1/models/{name}   # Get model details
```

## Project Structure

```
local-ai-backend/
├── ai/
│   ├── models.py           # Django models (GGUFModel)
│   ├── schemas/            # Pydantic schemas
│   ├── services/           # Business logic
│   │   ├── model_registry.py   # Model download/list
│   │   ├── model_loader.py     # LLM loading/inference
│   │   ├── chat_service.py    # Chat completions
│   │   └── embeddings_service.py
│   └── views.py            # API endpoints
├── core/
│   ├── settings.py         # Django configuration
│   ├── urls.py             # URL routing
│   └── wsgi.py / asgi.py
├── models/                 # GGUF model files
├── tests/
│   └── models/             # Utility scripts
│       ├── download.py
│       └── list_models.py
└── manage.py
```

## Configuration

Django settings are in `core/settings.py`. The project uses `django-environ` to load environment variables from `.env`.

Key settings:
- `BASE_DIR`: Project root (auto-detected)
- `MODELS_DIR`: Where GGUF files are stored (defaults to `./models`)
- `HF_TOKEN`: HuggingFace authentication token

## Development

### Running Tests

```bash
poetry run python manage.py test
```

### Database Migrations

```bash
poetry run python manage.py makemigrations
poetry run python manage.py migrate
```

## License

MIT