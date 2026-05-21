import gc
import logging
from pathlib import Path
from typing import Optional, Any
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from django.conf import settings


logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._models_dir = getattr(settings, "MODELS_DIR", "models")

    def download_model(
        self,
        repo_id: str,
        filename: str,
    ) -> str:
        def _get_hf_token() -> str:
            token = getattr(settings, "HF_TOKEN", "")
            if not token:
                logger.warning("HF_TOKEN is not set in settings, download speed may be slow and you may hit rate limits.")
            return token
        """
        Download a model from HuggingFace and save it to the models directory.

        Args:
            repo_id: HuggingFace repo ID (e.g., 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF')
            filename: Name of the GGUF file to download

        Returns:
            Path to the downloaded model file
        """
        logger.info(f"Downloading '{filename}' from '{repo_id}'...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(self._get_model_dir_path()),
            local_dir_use_symlinks=False,
            token=_get_hf_token(),
        )
        logger.info(f"Model downloaded to '{local_path}'")
        return local_path

    def list_models(self) -> list[str]:
        """List available GGUF model files in the models directory."""
        models_path = self._get_model_dir_path()
        return sorted(f.name for f in models_path.glob("*.gguf"))

    def get_model_path(self, model_name: str) -> str:
        """Get the path to the model file."""
        model_path = self._get_model_dir_path() / model_name
        if model_path.is_file():
            return str(model_path)
        raise FileNotFoundError(f"Model '{model_name}' not found in '{self._models_dir}'")

    def _get_model_dir_path(self) -> Path:
        """Get the path to the models directory."""
        models_path = Path(self._models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path
    

class ModelLoader:
    def __init__(self) -> None:
        self._loaded_model = None
        self._loaded_model_path = None

    @property
    def loaded_model(self) -> Optional[Llama]:
        return self._loaded_model

    @property
    def loaded_model_path(self) -> Optional[str]:
        return self._loaded_model_path

    def load_model(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> None:
        """Load a model by path"""
        self._loaded_model_path = model_path
        self._loaded_model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            seed=seed,
            **kwargs,
        )

    def clear_loaded_model(self) -> None:
        """Unload the currently loaded model from memory."""
        del self._loaded_model
        gc.collect()
        self._loaded_model = None