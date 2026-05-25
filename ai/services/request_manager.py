import threading
import logging
from typing import Iterator, Union, Any
from llama_cpp import ChatCompletion, ChatCompletionChunk, CreateCompletionResponse, CreateCompletionStreamResponse, CreateEmbeddingResponse

from ai.exceptions import ModelNotFoundError, APIError, ServiceError
from ai.services.llama_service import LlamaService, LlamaCppChatCompletionInput, LlamaCppCompletionInput, LlamaCppEmbeddingInput
from ai.services.model_service import ModelService

logger = logging.getLogger(__name__)


class RequestManager:
    def __init__(self, engine: LlamaService, model_service: ModelService) -> None:
        self.engine = engine
        self.model_service = model_service
        self._lock = threading.Lock()

    def _prepare_model(self, model_name: str, **loader_kwargs: Any) -> None:
        """
        Internal utility to check VRAM state, unload the old model if necessary,
        and load the requested model before execution.
        
        Must always be called from within a thread lock.
        
        NOTE: If load_ai_model fails, the engine may be left in a transitional state.
        This is acceptable in a local environment—the next request will re-evaluate
        and attempt to load again. For production, consider adding recovery logic.
        """
        target_path = self.model_service.get_ai_model_path(model_name)
        if target_path is None:
            raise ModelNotFoundError(f"Model '{model_name}' not found")

        current_path = self.engine.loaded_ai_model_path

        if current_path != target_path:
            # If a different model is taking up VRAM, eject it completely
            if current_path is not None:
                self.engine.unload_ai_model()

            # Merge DB-stored config with per-request overrides (explicit wins)
            db_config = self.model_service.get_loader_kwargs(model_name)
            merged = {**db_config, **loader_kwargs}
            self.engine.load_ai_model(ai_model_path=target_path, **merged)

    def execute_chat(
        self, 
        model_name: str, 
        input_data: LlamaCppChatCompletionInput, 
        loader_kwargs: Any = None
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """
        Thread-safe entrypoint for Chat Completions. 
        Blocks concurrent requests and manages VRAM access for both standard and streaming outputs.
        
        This method:
        1. Acquires a global lock (blocks if GPU is busy)
        2. Loads/unloads models as needed
        3. Executes the inference
        4. Manages lock release based on stream mode
        
        Args:
            model_name: Name of the model to load (without .gguf extension)
            input_data: Structured request data with model parameters
            loader_kwargs: Optional kwargs to pass to model loader
            
        Returns:
            ChatCompletion response or Iterator of ChatCompletionChunk for streaming
            
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model loading fails
        """
        loader_opts = loader_kwargs or {}
        thread_id = threading.get_ident()
        
        logger.info(f"[Thread {thread_id}] Attempting to acquire VRAM lock for model: {model_name}")
        
        # TODO: Consider adding a timeout here to prevent infinite user queuing
        # Example: if not self._lock.acquire(timeout=30): raise TimeoutError(...)
        self._lock.acquire()
        
        lock_released = False
        try:
            logger.info(f"[Thread {thread_id}] Lock acquired. Preparing model: {model_name}")
            self._prepare_model(model_name, **loader_opts)
            
            # Execute the inference
            result = self.engine.create_chat_completion(input_data)
            
            # FIXED: Trust the schema definition (input_data.stream), not runtime class type
            if getattr(input_data, 'stream', False):
                logger.info(f"[Thread {thread_id}] Streaming requested. Passing lock control to generator.")
                return self._stream_generator_wrapper(result, thread_id)
            else:
                # Standard mode: Lock is completely done, release immediately
                self._lock.release()
                lock_released = True
                logger.info(f"[Thread {thread_id}] Inference complete. Lock released.")
                return result

        except Exception as e:
            logger.error(f"[Thread {thread_id}] Execution failed: {str(e)}", exc_info=True)
            if not lock_released:
                self._lock.release()
                logger.info(f"[Thread {thread_id}] Lock released due to exception.")
            if isinstance(e, APIError):
                raise
            raise ServiceError(str(e)) from e

    def execute_completion(
        self, 
        model_name: str, 
        input_data: LlamaCppCompletionInput, 
        loader_kwargs: Any = None
    ) -> Union[CreateCompletionResponse, Iterator[CreateCompletionStreamResponse]]:
        """
        Thread-safe entrypoint for raw text/legacy completions.
        
        This method follows the same lock + model loading pattern as execute_chat,
        but calls the completion endpoint instead of chat completion.
        """
        loader_opts = loader_kwargs or {}
        thread_id = threading.get_ident()
        
        logger.info(f"[Thread {thread_id}] Attempting to acquire VRAM lock for model: {model_name}")
        
        self._lock.acquire()
        
        lock_released = False
        try:
            logger.info(f"[Thread {thread_id}] Lock acquired. Preparing model: {model_name}")
            self._prepare_model(model_name, **loader_opts)
            
            result = self.engine.create_completion(input_data)
            
            # FIXED: Use schema definition instead of runtime type check
            if getattr(input_data, 'stream', False):
                logger.info(f"[Thread {thread_id}] Streaming requested. Passing lock control to generator.")
                return self._stream_generator_wrapper(result, thread_id)
            else:
                self._lock.release()
                lock_released = True
                logger.info(f"[Thread {thread_id}] Inference complete. Lock released.")
                return result
                
        except Exception as e:
            logger.error(f"[Thread {thread_id}] Execution failed: {str(e)}", exc_info=True)
            if not lock_released:
                self._lock.release()
                logger.info(f"[Thread {thread_id}] Lock released due to exception.")
            if isinstance(e, APIError):
                raise
            raise ServiceError(str(e)) from e

    def execute_embedding(
        self,
        model_name: str,
        input_data: LlamaCppEmbeddingInput,
        loader_kwargs: Any = None,
    ) -> CreateEmbeddingResponse:
        """
        Thread-safe entrypoint for embeddings.
        Acquires the global lock, loads/unloads models as needed, executes inference.
        """
        loader_opts = loader_kwargs or {}
        thread_id = threading.get_ident()

        logger.info(f"[Thread {thread_id}] Attempting to acquire VRAM lock for model: {model_name}")

        self._lock.acquire()

        lock_released = False
        try:
            logger.info(f"[Thread {thread_id}] Lock acquired. Preparing model: {model_name}")
            self._prepare_model(model_name, **loader_opts)

            result = self.engine.create_embedding(input_data)

            self._lock.release()
            lock_released = True
            logger.info(f"[Thread {thread_id}] Embedding complete. Lock released.")
            return result

        except Exception as e:
            logger.error(f"[Thread {thread_id}] Embedding failed: {str(e)}", exc_info=True)
            if not lock_released:
                self._lock.release()
                logger.info(f"[Thread {thread_id}] Lock released due to exception.")
            if isinstance(e, APIError):
                raise
            raise ServiceError(str(e)) from e

    def _stream_generator_wrapper(self, stream: Iterator[Any], thread_id: int) -> Iterator[Any]:
        """
        A generator wrapper that keeps the thread lock active for the full duration 
        of the stream consumption. Releases only when the client has received the last token
        or disconnected abruptly.
        
        This wrapper ensures the lock is released even if:
        - The client closes the connection mid-stream
        - An exception occurs during chunk iteration
        - The stream is exhausted normally
        
        Args:
            stream: Iterator yielding response chunks
            thread_id: Thread ID for logging purposes
        """
        try:
            for chunk in stream:
                yield chunk
        finally:
            # The 'finally' block is guaranteed to execute even if the client closes the browser,
            # ensuring our thread lock is never left hanging.
            self._lock.release()
            logger.info(f"[Thread {thread_id}] Streaming complete or disconnected. Lock released.")