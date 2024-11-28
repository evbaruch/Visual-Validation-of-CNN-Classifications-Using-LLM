from LmmApi.LLMStrategy import LLMStrategy


# Concrete Strategy: OpenAI GPT
class chatGpt4o(LLMStrategy):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def generate_response(self, prompt: str, image: str) -> str:
        # Mocking API call; replace with OpenAI API call logic
        return f"OpenAI response for prompt: {prompt}"