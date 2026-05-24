# Interface for a request to be executed via the method execute()

from abc import ABC, abstractmethod
from ai.schemas.request import BaseRequest

class Request(ABC):
    def __init__(self, request_id: str, data: BaseRequest):
        self.request_id = request_id
        self.data = data

    @abstractmethod
    def execute(self):
        raise NotImplementedError("Subclasses must implement the execute method")