import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from django.conf import settings


logger = logging.getLogger(__name__)


UNSUPPORTED_PARAMETERS = {
    "n", "stream_options", "user", "parallel_tool_calls",
    "metadata", "service_tier", "reasoning_effort", "store",
}

PARAMETER_MAPPINGS = {
    "max_completion_tokens": "max_tokens",
}

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
        
        # TODO: Create a specific exception for this case and handle it in the caller
        raise FileNotFoundError(
            f"AI model '{ai_model_name}' not found in '{self._ai_models_dir}'. "
            f"Available models: {', '.join(self.list_ai_models())}"
        )

    def _get_ai_models_dir_path(self) -> Path:
        """Get the path to the AI models directory."""
        ai_models_path = Path(self._ai_models_dir)
        ai_models_path.mkdir(parents=True, exist_ok=True)
        return ai_models_path