from abc import ABC, abstractmethod

# Strategy Interface
class LLMStrategy(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, image: str) -> str:
        """Generate a response from the LLM based on the given prompt."""
        pass