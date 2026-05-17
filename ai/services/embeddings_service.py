import logging

from ai.schemas.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingObject,
    EmbeddingUsage,
)
from ai.models import GGUFModel

from .model_loader import model_loader

logger = logging.getLogger(__name__)


def create_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    model_name = request.model

    try:
        gguf_model = GGUFModel.objects.get(filename=model_name)
        if gguf_model.type != GGUFModel.ModelType.EMBEDDING:
            raise ValueError(f"Model '{model_name}' is not an embedding model")
    except GGUFModel.DoesNotExist:
        raise ValueError(f"Model '{model_name}' not found in database")

    if not model_loader.loaded_model or model_loader.loaded_model_name != model_name:
        model_loader.load_model(model_name, embedding=True)

    llm = model_loader.loaded_model
    if llm is None:
        raise RuntimeError(f"Model '{model_name}' failed to load")

    input_texts = request.input
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    embedding_objects = []
    total_tokens = 0

    for idx, text in enumerate(input_texts):
        result = llm.create_embedding(text)
        embedding = result["data"][0]["embedding"]
        embedding_objects.append(
            EmbeddingObject(
                object="embedding",
                index=idx,
                embedding=embedding,
            )
        )
        total_tokens += result.get("usage", {}).get("prompt_tokens", 0)

    return EmbeddingResponse(
        object="list",
        data=embedding_objects,
        model=model_name,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )