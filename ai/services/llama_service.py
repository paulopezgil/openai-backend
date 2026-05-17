import logging
from typing import Optional, Dict, Any, List

from llama_cpp import Llama

from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class LlamaService:
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self._model_manager = model_manager or ModelManager()

    @property
    def is_model_loaded(self) -> bool:
        return self._model_manager.loaded_model is not None

    @property
    def loaded_model_name(self) -> Optional[str]:
        return self._model_manager.loaded_model_name

    def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a text completion using the specified model.

        Args:
            prompt: Input text prompt
            model: Model name to use (will load if not already loaded)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repeat_penalty: Penalty for repeated tokens
            stop: List of stop sequences
            stream: Whether to stream the response

        Returns:
            Dict with 'text', 'stop_reason', 'model', and token usage info
        """
        if self._model_manager.loaded_model is None or self._model_manager.loaded_model_name != model:
            logger.info(f"Loading model '{model}' for completion")
            self._model_manager.load_model(model_name=model)

        model_instance = self._model_manager.loaded_model

        if stop is None:
            stop = []

        logger.info(f"Creating completion (model={model}, max_tokens={max_tokens}, temp={temperature})")

        if stream:
            raise NotImplementedError("Streaming not yet implemented")

        response = model_instance(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
            echo=False,
        )

        return {
            "text": response["choices"][0]["text"].strip(),
            "stop_reason": response["choices"][0].get("finish_reason", "unknown"),
            "model": model,
            "prompt_tokens": response.get("prompt_tokens", 0),
            "completion_tokens": response.get("completion_tokens", 0),
            "total_tokens": response.get("total_tokens", 0),
        }

    def load_model(
        self,
        model_name: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        seed: int = -1,
    ) -> None:
        """Load a model by name."""
        self._model_manager.load_model(
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=seed,
        )

    def clear_model(self) -> None:
        """Unload the current model."""
        self._model_manager.clear_loaded_model()

    def list_models(self) -> list[str]:
        """List available models."""
        return self._model_manager.list_models()


llama_service = LlamaService()