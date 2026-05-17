import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ModelRegistry:
    _instance: Optional["ModelRegistry"] = None

    def __new__(cls, models_dir: str = "./models"):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._models_dir = models_dir
        return cls._instance

    def __init__(self, models_dir: str = "./models") -> None:
        if hasattr(self, '_models_dir') and self._models_dir != models_dir:
            raise ValueError(
                f"models_dir is already set to '{self._models_dir}', "
                f"cannot change to '{models_dir}'"
            )

    def download_model(
        self,
        repo_id: str,
        filename: str,
    ) -> str:
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
            local_dir=str(self._create_model_dir_path()),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to '{local_path}'")
        return local_path

    def list_models(self) -> list[str]:
        """List available GGUF model files in the models directory."""
        models_path = self._create_model_dir_path()
        return sorted(f.name for f in models_path.glob("*.gguf"))

    def create_model_path(self, model_name: str) -> str:
        """Create the path to the model file."""
        model_path = self._create_model_dir_path() / model_name
        if model_path.is_file():
            return str(model_path)
        raise FileNotFoundError(f"Model '{model_name}' not found in '{self._models_dir}'")

    def _create_model_dir_path(self) -> Path:
        """Create the path to the models directory."""
        models_path = Path(self._models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path


model_registry = ModelRegistry()