import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from threading import RLock

from llama_cpp import Llama
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class ModelManager:
    # Note: These are declared as class attributes for type hints only.
    # They are actually set as instance attributes in __new__ to implement
    # a singleton pattern (one instance globally).
    _instance: Optional["ModelManager"] = None
    _loaded_model: Optional[Llama]
    _loaded_model_name: Optional[str]
    _models_dir: str
    _lock: RLock

    def __new__(cls, models_dir: str = "./models"):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._loaded_model = None
            cls._instance._loaded_model_name = None
            cls._instance._models_dir = models_dir
            cls._instance._lock = RLock()
        return cls._instance

    def __init__(self, models_dir: str = "./models") -> None:
        if hasattr(self, '_models_dir') and self._models_dir != models_dir:
            raise ValueError(
                f"models_dir is already set to '{self._models_dir}', "
                f"cannot change to '{models_dir}'"
            )

    @property
    def loaded_model(self) -> Optional[Llama]:
        return self._loaded_model

    @property
    def loaded_model_name(self) -> Optional[str]:
        return self._loaded_model_name

    def load_model(
        self,
        model_name: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        seed: int = -1,
        **kwargs: Any,
    ) -> None:
        """ Load a model by name, unloading any currently loaded model if it's different."""
        
        with self._lock:
            if self._loaded_model_name != model_name:
                # Clear the currently loaded model if it's different from the one we want to load
                self.clear_loaded_model()

                # Find the actual model file path (the largest .gguf file in the model directory)
                model_path = self._create_model_path(model_name)

                # Load the model using Llama and log the process
                logger.info(f"Loading model '{model_name}' from {model_path}...")
                self._loaded_model = Llama(
                    model_path=model_path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    n_threads=n_threads,
                    seed=seed,
                    **kwargs,
                )
                logger.info(f"Model '{model_name}' loaded successfully")
                self._loaded_model_name = model_name
        
    def clear_loaded_model(self) -> None:
        """Unload the currently loaded model from memory."""

        with self._lock:
            if self._loaded_model is not None:
                # Log the model being cleared for better visibility in logs
                logger.info(f"Cleaning '{self._loaded_model_name}' from memory")   
                
                # Explicitly delete the loaded model
                del self._loaded_model
                
                # Force garbage collection to free up memory immediately
                gc.collect()

                # Clear the loaded model info after deletion
                self._loaded_model = None
                self._loaded_model_name = None

    def list_available_models(self) -> list[str]:
        """List available GGUF model files in the models directory."""
        models_path = self._create_model_dir_path()
        return sorted(f.name for f in models_path.glob("*.gguf")) 

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

    def _create_model_dir_path(self) -> Path:
        """Create the path to the models directory."""
        models_path = Path(self._models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        return models_path
        
    def _create_model_path(self, model_name: str) -> str:
        """Create the path to the model file."""
        model_path = self._create_model_dir_path() / model_name
        if model_path.is_file():
            return str(model_path)
        raise FileNotFoundError(f"Model '{model_name}' not found in '{self._models_dir}'")