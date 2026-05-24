# Class to implement a completion request inheriting from the base Request class

from ai.queue.requests.request import Request
from ai.services.ai_model_service import AIModelRegistry, AIModelLoader, call_ai_model
from ai.schemas.chat import ChatCompletionRequest

class CompletionRequest(Request):
    def __init__(self, request_id: str, data: ChatCompletionRequest):
        super().__init__(request_id, data)
        self.data: ChatCompletionRequest = self.data

    def execute(self):
        modelRegistry = AIModelRegistry()
        modelLoader = AIModelLoader()
        
        # Get the AI model path based on the model name in the request data
        try:
            ai_model_path = modelRegistry.get_ai_model_path(self.data.model)
        except FileNotFoundError:
            # TODO: Create a specific exception for this case and handle it in the caller
            raise FileNotFoundError(f"AI model '{self.data.model}' not found. Please download it first.") 
        
        # Check current loaded model and load the requested one if it's different
        if modelLoader.loaded_ai_model_path != ai_model_path:
            modelLoader.clear_loaded_ai_model()
            modelLoader.load_ai_model(ai_model_path)

        # Call the AI model with the request data and return the result
        return call_ai_model(modelLoader.loaded_ai_model, **self.data.model_dump())         