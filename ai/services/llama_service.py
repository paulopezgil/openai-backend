import gc
from typing import Optional, Any, Union, Iterator, Dict
from pydantic import BaseModel
from llama_cpp import (
    Llama, ChatCompletion, ChatCompletionChunk,
    ChatCompletionRequestMessage, 
)



class LlamaCppChatCompletionInput(BaseModel):
    # Required parameters
    model: str
    messages: list[ChatCompletionRequestMessage]

    # Llama-cpp supported parameters (Optional with defaults)
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, list[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[list[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None

    # Advanced llama-cpp sampling parameters
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    tfs_z: Optional[float] = None
    mirostat_mode: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None


class LlamaService:
    _instance = None

    def __new__(cls) -> "LlamaService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._loaded_ai_model = None
        self._loaded_ai_model_path = None
        self._initialized = True

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

    def call_loaded_ai_model(self, input: LlamaCppChatCompletionInput) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """Call the AI model with the given parameters, after validating and preparing them."""
        
        # TODO: Use custom exception types for better error handling in the caller
        if self._loaded_ai_model is None:
            raise ValueError("No AI model is currently loaded. Please load a model before calling it.")

        return self._loaded_ai_model.create_chat_completion(input.model_dump())