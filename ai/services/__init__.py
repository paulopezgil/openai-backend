from .llama_service import LlamaService
from .model_service import ModelService
from .request_manager import RequestManager

request_manager = RequestManager(
    engine=LlamaService(),
    model_service=ModelService(),
)
