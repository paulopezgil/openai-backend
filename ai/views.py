import json
import logging

from django.http import StreamingHttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from ai.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from ai.schemas.embeddings import EmbeddingRequest
from ai.schemas.models import ModelListResponse, Model

from .services import chat_service, embeddings_service
from .models import GGUFModel

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class ChatCompletionsView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=400,
            )

        try:
            chat_request = ChatCompletionRequest(**data)
        except Exception as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=422,
            )

        return self._handle_completion(chat_request)

    def _handle_completion(self, chat_request: ChatCompletionRequest):
        try:
            if chat_request.stream:
                return self._stream_response(chat_request)
            response = chat_service.create_chat_completion(chat_request)
            return JsonResponse(response.model_dump(mode="json"))
        except FileNotFoundError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=400,
            )
        except RuntimeError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "internal_error"}},
                status=500,
            )
        except Exception as e:
            logger.exception("Error creating chat completion")
            return JsonResponse(
                {"error": {"message": str(e), "type": "internal_error"}},
                status=500,
            )

    def _stream_response(self, chat_request: ChatCompletionRequest):
        generator = chat_service.create_chat_completion_stream(chat_request)

        def event_stream():
            for chunk in generator:
                yield f"data: {json.dumps(chunk.model_dump(mode='json'))}\n\n"
            yield "data: [DONE]\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type="text/event-stream",
        )
        response["Cache-Control"] = "no-cache"
        response["Connection"] = "keep-alive"
        return response


@method_decorator(csrf_exempt, name="dispatch")
class EmbeddingsView(View):
    def post(self, request):
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=400,
            )

        try:
            embed_request = EmbeddingRequest(**data)
        except Exception as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=422,
            )

        try:
            response = embeddings_service.create_embedding(embed_request)
            return JsonResponse(response.model_dump(mode="json"))
        except FileNotFoundError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status=400,
            )
        except RuntimeError as e:
            return JsonResponse(
                {"error": {"message": str(e), "type": "internal_error"}},
                status=500,
            )
        except Exception as e:
            logger.exception("Error creating embedding")
            return JsonResponse(
                {"error": {"message": str(e), "type": "internal_error"}},
                status=500,
            )


class ModelsListView(View):
    def get(self, request):
        models = GGUFModel.objects.all().order_by("-created")
        return JsonResponse(
            ModelListResponse(
                object="list",
                data=[
                    Model(
                        id=m.filename,
                        object="model",
                        created=int(m.created.timestamp()),
                        owned_by="local",
                    )
                    for m in models
                ],
            ).model_dump(mode="json")
        )


class ModelDetailView(View):
    def get(self, request, model_name: str):
        try:
            gguf_model = GGUFModel.objects.get(filename=model_name)
        except GGUFModel.DoesNotExist:
            return JsonResponse(
                {"error": {"message": f"Model '{model_name}' not found", "type": "invalid_request_error"}},
                status=404,
            )

        return JsonResponse(
            Model(
                id=gguf_model.filename,
                object="model",
                created=int(gguf_model.created.timestamp()),
                owned_by="local",
            ).model_dump(mode="json")
        )