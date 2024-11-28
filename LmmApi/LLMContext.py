import LLMStrategy


# Context Class
class LLMContext:
    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: LLMStrategy):
        """Set a different strategy at runtime."""
        self.strategy = strategy

    def get_response(self, prompt: str) -> str:
        """Generate a response using the current strategy."""
        return self.strategy.generate_response(prompt)