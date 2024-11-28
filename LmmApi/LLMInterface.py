from LmmApi.LLMStrategy import LLMStrategy


# Context Class
class LLMInterface:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: LLMStrategy):
        """Set a different strategy at runtime."""
        self.strategy = strategy

    def get_response(self, prompt: str, image: str) -> str:
        """Generate a response using the current strategy."""
        return self.strategy.generate_response(prompt = prompt, image = image)