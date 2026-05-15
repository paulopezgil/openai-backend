import json
from django.http import StreamingHttpResponse
from ninja import NinjaAPI
from .models import AIModel
from services.manager import VulkanModelManager
from .schemas import ChatCompletionRequest

api = NinjaAPI(title="Local Vulkan AI API", version="1.0.0")

async def sse_generator(llm, payload):
    """Generates Server-Sent Events for OpenAI streaming"""
    # Note: Create completion is typically a blocking call in llama-cpp, 
    # but we yield from its internal generator.
    iterator = llm.create_chat_completion(**payload, stream=True)
    
    for chunk in iterator:
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

@api.get("/v1/models")
def list_models(request):
    models = AIModel.objects.all()
    return {
        "object": "list",
        "data": [{"id": m.name, "object": "model", "owned_by": "local"} for m in models]
    }

@api.post("/v1/chat/completions")
def chat_completions(request, data: ChatCompletionRequest):
    try:
        model_db = AIModel.objects.get(name=data.model)
    except AIModel.DoesNotExist:
        return {"error": "Model not found or inactive"}, 404

    # Load/Switch model in VRAM
    llm = VulkanModelManager.get_llm(
        model_db.name, 
        model_db.local_path, 
        model_db.context_size, 
        model_db.n_gpu_layers
    )

    payload = {
        "messages": [m.dict() for m in data.messages],
        "temperature": data.temperature,
        "max_tokens": data.max_tokens,
    }

    if data.stream:
        return StreamingHttpResponse(
            sse_generator(llm, payload),
            content_type="text/event-stream"
        )

    return llm.create_chat_completion(**payload, stream=False)