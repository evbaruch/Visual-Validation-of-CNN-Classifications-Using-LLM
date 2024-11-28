import LLMStrategy
import ollama

# Concrete Strategy: OpenAI GPT 
class llama32Vision11b(LLMStrategy):

    def __init__(self):
        self.model = 'llama3.2-vision:11b'


    def generate_response(self, prompt: str, image: str) -> str:
        response = ollama.chat(model=self.model, messages=[
            { 
                'role': 'user',
                'content': prompt,
                'images': [image]
            },
        ])

        return f"llama32Vision11b response for prompt: {prompt}"