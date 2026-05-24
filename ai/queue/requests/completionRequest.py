# Class to implement a completion request inheriting from the base Request class
from .request import Request
from ai.services.ai_model_service import AIModelRegistry, AIModelLoader

from ai.schemas.chat import ChatCompletionRequest

class CompletionRequest(Request):
    def __init__(self, request_id: str, data: ChatCompletionRequest):
        super().__init__(request_id, data)
        self.chat_request = data

    def execute(self):
        modelRegistry = AIModelRegistry()
        modelLoader = AIModelLoader()
        
        # Ensure the AI model is downloaded and loaded
        ai_model_name = self.chat_request.model
        try:
            ai_model_path = modelRegistry.get_ai_model_path(ai_model_name)
        except FileNotFoundError:
            # TODO: Create a specific exception for this case and handle it in the caller
            raise FileNotFoundError(f"AI model '{ai_model_name}' not found. Please download it first.") 
        
        # Check current loaded model and load the requested one if it's different
        if modelLoader.loaded_ai_model_path != ai_model_path:
            modelLoader.clear_loaded_ai_model()
            modelLoader.load_ai_model(ai_model_path)
            
        
        
        