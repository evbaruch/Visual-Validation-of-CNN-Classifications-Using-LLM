from LmmApi.LLMStrategy import LLMStrategy
import ollama

# Concrete Strategy: OpenAI GPT 
class llama32Vision11b(LLMStrategy):

    def __init__(self):
        self.model = 'llama3.2-vision:11b'


    def generate_response(self, prompt: str, image: str) -> str:
        
        try:
            response = ollama.chat(model=self.model, messages=[
                { 
                    'role': 'user',
                    'content': prompt,
                    'images': [image]
                },
            ])
            return response['message']['content']
             
        except TypeError as e:
            print("LmmApi\llama32Vision11b.py: Error in ollama.chat:", e)
            raise