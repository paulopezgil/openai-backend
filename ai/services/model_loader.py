import gc
import logging
from typing import Optional, Dict, Any
from threading import RLock

from llama_cpp import Llama

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class ModelLoader:
    _instance: Optional["ModelLoader"] = None

    def __new__(cls, model_registry: Optional[ModelRegistry] = None):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._loaded_model = None
            cls._instance._loaded_model_name = None
            cls._instance._lock = RLock()
            cls._instance._model_registry = model_registry or ModelRegistry()
        return cls._instance

    def __init__(self, model_registry: Optional[ModelRegistry] = None) -> None:
        pass

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
        """Load a model by name, unloading any currently loaded model if it's different."""
        with self._lock:
            if self._loaded_model_name != model_name:
                self.clear_loaded_model()

                model_path = self._model_registry.create_model_path(model_name)

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
                logger.info(f"Cleaning '{self._loaded_model_name}' from memory")
                del self._loaded_model
                gc.collect()
                self._loaded_model = None
                self._loaded_model_name = None


model_loader = ModelLoader()