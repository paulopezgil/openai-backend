# Local AI Backend

A OpenAI-compatible Django-based backend that uses llama-cpp-python to run local AI models.

## Features

- **Chat Completions**: OpenAI-compatible chat endpoint with streaming support
- **Embeddings**: OpenAI-compatible embeddings endpoint
- **Model Management**: REST API for listing and managing local GGUF models
- **Database-backed Model Registry**: Stores model metadata in Django models
- **CSRF Exempt**: Designed for API usage (adjust as needed for production)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd local-ai-backend

# Install dependencies
poetry install

# Apply migrations
poetry run python manage.py migrate

# Start the development server
poetry run python manage.py runserver 0.0.0.0:8000
```

## API Endpoints

### Chat Completions
```
POST /v1/chat/completions
```
OpenAI-compatible chat completions endpoint with streaming support.

### Embeddings
```
POST /v1/embeddings
```
OpenAI-compatible embeddings endpoint.

### Model Management
```
GET /v1/models
```
List all available models.

```
GET /v1/models/{model_name}
```
Get details for a specific model.

## Model Storage

Models are stored in the `./models/` directory as GGUF files. Model metadata is stored in the Django database via the `GGUFModel` model.

## Configuration

See `core/settings.py` for Django configuration. Adjust as needed for your environment.

## Usage Examples

### Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Embeddings
```bash
curl -X POST http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-embedding-model.gguf",
    "input": "Your text here"
  }'
```

## Development

This project uses:
- Django 5.0+
- llama-cpp-python
- Poetry for dependency management

Run migrations when modifying models:
```bash
poetry run python manage.py makemigrations
poetry run python manage.py migrate
```