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
    pass

@method_decorator(csrf_exempt, name="dispatch")
class EmbeddingsView(View):
    pass

class ModelsListView(View):
    pass