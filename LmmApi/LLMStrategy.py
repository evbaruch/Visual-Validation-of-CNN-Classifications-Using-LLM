from abc import ABC, abstractmethod
from pydantic import BaseModel

# Strategy Interface
class LLMStrategy(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, image: str , jsonDescription: BaseModel ) -> str:
        """Generate a response from the LLM based on the given prompt."""
        pass

