import logging
from typing import Iterator

from ai.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
)

from .model_loader import model_loader

logger = logging.getLogger(__name__)


def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    model_name = request.model

    if not model_loader.loaded_model or model_loader.loaded_model_name != model_name:
        model_loader.load_model(model_name)

    llm = model_loader.loaded_model
    if llm is None:
        raise RuntimeError(f"Model '{model_name}' failed to load")

    params = request.model_dump(exclude={"model"})

    if request.max_completion_tokens:
        params["max_tokens"] = request.max_completion_tokens
    elif request.max_tokens:
        params["max_tokens"] = request.max_tokens

    params["stream"] = False

    result = llm.create_chat_completion(**params)

    return ChatCompletionResponse(
        id=result["id"],
        created=result["created"],
        model=result["model"],
        choices=[
            {
                "index": choice["index"],
                "message": {
                    "role": choice["message"]["role"],
                    "content": choice["message"].get("content"),
                    "tool_calls": choice["message"].get("tool_calls"),
                    "refusal": choice["message"].get("refusal"),
                    "annotations": [],
                },
                "finish_reason": choice.get("finish_reason"),
                "logprobs": choice.get("logprobs"),
            }
            for choice in result["choices"]
        ],
        usage=result["usage"],
    )


def create_chat_completion_stream(request: ChatCompletionRequest) -> Iterator[ChatCompletionChunk]:
    model_name = request.model

    if not model_loader.loaded_model or model_loader.loaded_model_name != model_name:
        model_loader.load_model(model_name)

    llm = model_loader.loaded_model
    if llm is None:
        raise RuntimeError(f"Model '{model_name}' failed to load")

    params = request.model_dump(exclude={"model"})

    if request.max_completion_tokens:
        params["max_tokens"] = request.max_completion_tokens
    elif request.max_tokens:
        params["max_tokens"] = request.max_tokens

    params["stream"] = True

    stream = llm.create_chat_completion(**params)

    for chunk in stream:
        yield ChatCompletionChunk(
            id=chunk["id"],
            created=chunk["created"],
            model=chunk["model"],
            choices=[
                {
                    "index": choice["index"],
                    "delta": {
                        "role": choice["delta"].get("role"),
                        "content": choice["delta"].get("content"),
                        "tool_calls": choice["delta"].get("tool_calls"),
                        "refusal": choice["delta"].get("refusal"),
                    },
                    "finish_reason": choice.get("finish_reason"),
                    "logprobs": choice.get("logprobs"),
                }
                for choice in chunk["choices"]
            ],
            usage=chunk.get("usage"),
        )