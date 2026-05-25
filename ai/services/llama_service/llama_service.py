import gc
from typing import Optional, Any, Union, Iterator
from llama_cpp import (
    Llama, ChatCompletion, ChatCompletionChunk,
    CreateCompletionResponse, CreateCompletionStreamResponse,
    CreateEmbeddingResponse
)
from ai.exceptions import ModelNotLoadedError
from .schemas import (
    LlamaCppChatCompletionInput,
    LlamaCppCompletionInput,
    LlamaCppEmbeddingInput,
)


class LlamaService:
    def __init__(self) -> None:
        self._loaded_ai_model = None
        self._loaded_ai_model_path = None

    @property
    def loaded_ai_model(self) -> Optional[Llama]:
        return self._loaded_ai_model

    @property
    def loaded_ai_model_path(self) -> Optional[str]:
        return self._loaded_ai_model_path

    def load_ai_model(
        self,
        ai_model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> None:
        """Load an AI model by path"""
        self._loaded_ai_model_path = ai_model_path
        self._loaded_ai_model = Llama(
            model_path=ai_model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=seed,
            **kwargs,
        )

    def unload_ai_model(self) -> None:
        """Unload the currently loaded AI model from memory."""
        del self._loaded_ai_model
        gc.collect()
        self._loaded_ai_model = None
        self._loaded_ai_model_path = None

    def create_chat_completion(self, input: LlamaCppChatCompletionInput) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Call the AI model with the given parameters, after validating and preparing them."""
        
        if self._loaded_ai_model is None:
            raise ModelNotLoadedError("No AI model is currently loaded. Please load a model before calling it.")

        return self._loaded_ai_model.create_chat_completion(
            **input.model_dump(exclude_none=True)
        )

    def create_completion(self, input: LlamaCppCompletionInput) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """Call the AI model with the given parameters, after validating and preparing them."""

        if self._loaded_ai_model is None:
            raise ModelNotLoadedError("No AI model is currently loaded. Please load a model before calling it.")

        return self._loaded_ai_model.create_completion(
            **input.model_dump(exclude_none=True)
        )

    def create_embedding(self, input: LlamaCppEmbeddingInput) -> CreateEmbeddingResponse:
        """Call the AI model to create embeddings with the given parameters."""

        if self._loaded_ai_model is None:
            raise ModelNotLoadedError("No AI model is currently loaded. Please load a model before calling it.")

        return self._loaded_ai_model.create_embedding(
            **input.model_dump(exclude_none=True)
        )

    def __del__(self):
        """Ensure AI model is unloaded when the service is destroyed."""
        if self._loaded_ai_model is not None:
            self.unload_ai_model()
