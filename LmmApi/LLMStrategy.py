from abc import ABC, abstractmethod
from pydantic import BaseModel
from logger import setup_logger, log_function_call, log_class_methods


# Strategy Interface
@log_class_methods
class LLMStrategy(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, image: str , jsonDescription: BaseModel ) -> str:
        """Generate a response from the LLM based on the given prompt."""
        pass

