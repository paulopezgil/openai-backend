import logging
from pathlib import Path
from typing import Any
from huggingface_hub import hf_hub_download
from django.conf import settings
from ai.models import GGUFModel


logger = logging.getLogger(__name__)


class ModelService:
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
        """Download an AI model from HuggingFace and save it to the AI models directory."""

        def _get_hf_token() -> str:
            token = getattr(settings, "HF_TOKEN", "")
            if not token:
                logger.warning("HF_TOKEN is not set in settings, download speed may be slow and you may hit rate limits.")
            return token
        
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
        return None
    
    def ai_model_exists(self, ai_model_name: str) -> bool:
        """Check if the AI model file exists."""
        return self.get_ai_model_path(ai_model_name) is not None

    def get_loader_kwargs(self, ai_model_name: str) -> dict[str, Any]:
        """Return DB-stored loader params for a model, falling back to defaults."""
        try:
            config = GGUFModel.objects.get(filename=ai_model_name)
        except GGUFModel.DoesNotExist:
            try:
                config = GGUFModel.objects.get(filename=f"{ai_model_name}.gguf")
            except GGUFModel.DoesNotExist:
                return {}
        return {
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "n_threads": config.n_threads,
            "seed": config.seed,
        }

    def _get_ai_models_dir_path(self) -> Path:
        """Get the path to the AI models directory."""
        ai_models_path = Path(self._ai_models_dir)
        ai_models_path.mkdir(parents=True, exist_ok=True)
        return ai_models_path