import gc
import os
import threading
from llama_cpp import Llama

class VulkanModelManager:
    _instance = None
    _current_model_id = None
    _lock = threading.Lock()

    @classmethod
    def get_llm(cls, model_id, model_path, n_ctx, n_gpu_layers):
        with cls._lock:
            # Set Vulkan visibility for dual AMD GPUs
            os.environ["GGML_VK_VISIBLE_DEVICES"] = "0,1"

            if cls._current_model_id == model_id and cls._instance:
                return cls._instance

            # Unload to prevent VRAM overflow
            if cls._instance:
                print(f"Unloading {cls._current_model_id}...")
                cls._instance = None
                gc.collect()

            print(f"Loading {model_id} into Vulkan VRAM...")
            cls._instance = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=False,
                n_threads=os.cpu_count() or 4
            )
            cls._current_model_id = model_id
            return cls._instance