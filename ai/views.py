import json
import logging
import time
from typing import Union

from django.http import StreamingHttpResponse, JsonResponse, HttpRequest
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from openai.types.chat import ChatCompletion as ChatCompletionResponse
from openai.types import Completion as CompletionResponse
from openai.types import CreateEmbeddingResponse as EmbeddingResponse
from openai.types.model import Model

from ai.services import request_manager
from ai.services.llama_service import (
    LlamaCppChatCompletionInput,
    LlamaCppEmbeddingInput,
)
from ai.services.llama_service.cleaners import clean_chat_params, clean_embedding_params
from ai.exceptions import BadRequest, handle_exception

logger = logging.getLogger(__name__)


def _stream_response(result):
    try:
        for chunk in result:
            yield f"data: {chunk.model_dump_json()}\n\n"
    except GeneratorExit:
        pass
    except Exception as e:
        logger.exception("Streaming error")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        yield "data: [DONE]\n\n"


@method_decorator(csrf_exempt, name="dispatch")
class ChatCompletionsView(View):
    def post(self, request: HttpRequest) -> Union[JsonResponse, StreamingHttpResponse]:
        try:
            body = json.loads(request.body)
            cleaned = clean_chat_params(body)

            model = cleaned.get("model")
            if not model:
                raise BadRequest("model is required")

            try:
                input_data = LlamaCppChatCompletionInput(**cleaned)
            except Exception as exc:
                raise BadRequest(str(exc)) from exc
            is_stream = bool(cleaned.get("stream", False))

            result = request_manager.execute_chat(model, input_data)

        except Exception as e:
            return handle_exception(e)

        if is_stream:
            return StreamingHttpResponse(
                _stream_response(result),
                content_type="text/event-stream",
            )

        return JsonResponse(result.model_dump())


@method_decorator(csrf_exempt, name="dispatch")
class EmbeddingsView(View):
    def post(self, request: HttpRequest) -> JsonResponse:
        try:
            body = json.loads(request.body)
            cleaned = clean_embedding_params(body)

            model = cleaned.get("model")
            if not model:
                raise BadRequest("model is required")

            try:
                input_data = LlamaCppEmbeddingInput(**cleaned)
            except Exception as exc:
                raise BadRequest(str(exc)) from exc
            result = request_manager.execute_embedding(model, input_data)

        except Exception as e:
            return handle_exception(e)

        return JsonResponse(result.model_dump())


class ModelsView(View):
    def get(self, request: HttpRequest) -> JsonResponse:
        try:
            model_names = request_manager.model_service.list_ai_models()
        except Exception as e:
            return handle_exception(e)

        now = int(time.time())
        data = [
            Model(id=name, created=now, owned_by="local")
            for name in model_names
        ]

        return JsonResponse({
            "object": "list",
            "data": [m.model_dump() for m in data],
        })
