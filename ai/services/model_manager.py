import logging
from pathlib import Path
from typing import Optional, Dict, Any
from threading import RLock

from llama_cpp import Llama

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
        """
        Load a GGUF model into memory.

        Use the `loaded_model` property to access the loaded Llama instance.
        """
        with self._lock:
            if self.loaded_model is not None and self.loaded_model_name != model_name:
                self.clear_loaded_model()

            actual_model_path = self._find_model_path(model_name)

            logger.info(f"Loading model '{model_name}' from {actual_model_path}...")
            self._loaded_model = Llama(
                model_path=actual_model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                seed=seed,
                **kwargs,
            )
            logger.info(f"Model '{model_name}' loaded successfully")

            self._loaded_model_name = model_name
        
    def list_models(self) -> list[str]:
        models_path = Path(self._models_dir)
        if not models_path.exists():
            return []
        return [
            d.name
            for d in models_path.iterdir()
            if d.is_dir() and list(d.glob("*.gguf"))
        ]

    def clear_loaded_model(self) -> None:
        with self._lock:
            if self._loaded_model is not None:
                model_name = self._loaded_model_name
                del self._loaded_model
                self._loaded_model = None
                self._loaded_model_name = None
                logger.info(f"Model '{model_name}' cleared from memory")    

    def _find_model_path(self, model_name: str) -> str:
        model_path = Path(self._models_dir) / model_name
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"Model '{model_name}' not found in '{self._models_dir}'")

        gguf_files = sorted(
            model_path.glob("*.gguf"),
            key=lambda f: f.stat().st_size,
            reverse=True
        )
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF file found in: {model_path}")

        return str(gguf_files[0])