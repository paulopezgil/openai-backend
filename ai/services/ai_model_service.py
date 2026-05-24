import gc
import logging
from pathlib import Path
from typing import Optional, Any
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from django.conf import settings


logger = logging.getLogger(__name__)


UNSUPPORTED_PARAMETERS = {
    "n", "stream_options", "user", "parallel_tool_calls",
    "metadata", "service_tier", "reasoning_effort", "store",
}

PARAMETER_MAPPINGS = {
    "max_completion_tokens": "max_tokens",
}

def call_ai_model(ai_model: Llama, **kwargs: Any) -> dict:
    """Call the AI model with the given parameters."""

    def _prepare_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        if not kwargs.get("messages"):
            raise ValueError("'messages' parameter is required and cannot be empty")

        prepared = {}
        for k, v in kwargs.items():
            if v is None or k in UNSUPPORTED_PARAMETERS:
                continue
            mapped = PARAMETER_MAPPINGS.get(k, k)
            prepared[mapped] = v

        return prepared

    return ai_model.create_chat_completion(**_prepare_kwargs(kwargs))


class AIModelRegistry:
    def __init__(self) -> None:
        self._ai_models_dir = getattr(settings, "AI_MODELS_DIR", "ai_models")

    @staticmethod
    def _ensure_gguf_extension(ai_model_name: str) -> str:
        """Append .gguf extension to AI model name if not already present."""
        if not ai_model_name.endswith(".gguf"):
            return f"{ai_model_name}.gguf"
        return ai_model_name

    def download_ai_model(
        self,
        repo_id: str,
        filename: str,
    ) -> str:
        def _get_hf_token() -> str:
            token = getattr(settings, "HF_TOKEN", "")
            if not token:
                logger.warning("HF_TOKEN is not set in settings, download speed may be slow and you may hit rate limits.")
            return token
        """Download an AI model from HuggingFace and save it to the AI models directory."""
        logger.info(f"Downloading '{filename}' from '{repo_id}'...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self._get_ai_models_dir_path()),
            local_dir_use_symlinks=False,
            token=_get_hf_token(),
        )
        logger.info(f"AI model downloaded to '{local_path}'")
        return local_path

    def list_ai_models(self) -> list[str]:
        """List available GGUF AI model files in the AI models directory (without .gguf extension)."""
        ai_models_path = self._get_ai_models_dir_path()
        return sorted(f.stem for f in ai_models_path.glob("*.gguf"))

    def get_ai_model_path(self, ai_model_name: str) -> str:
        """Get the path to the AI model file. Accepts AI model names with or without .gguf extension."""
        # Ensure the AI model name has the .gguf extension
        full_ai_model_name = self._ensure_gguf_extension(ai_model_name)
        ai_model_path = self._get_ai_models_dir_path() / full_ai_model_name
        if ai_model_path.is_file():
            return str(ai_model_path)
        raise FileNotFoundError(f"AI model '{ai_model_name}' not found in '{self._ai_models_dir}'")

    def _get_ai_models_dir_path(self) -> Path:
        """Get the path to the AI models directory."""
        ai_models_path = Path(self._ai_models_dir)
        ai_models_path.mkdir(parents=True, exist_ok=True)
        return ai_models_path
    

class AIModelLoader:
    _instance = None

    def __new__(cls) -> "AIModelLoader":
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

    def clear_loaded_ai_model(self) -> None:
        """Unload the currently loaded AI model from memory."""
        del self._loaded_ai_model
        gc.collect()
        self._loaded_ai_model = None
        self._loaded_ai_model_path = None