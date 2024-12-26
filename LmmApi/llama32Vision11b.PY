from LmmApi.LLMStrategy import LLMStrategy
from data import global_data as gd
from ollama import chat
from pydantic import BaseModel
from typing import Literal

# Concrete Strategy: OpenAI GPT 
class llama32Vision11b(LLMStrategy):
    def __init__(self):
        self.model = 'llama3.2-vision:11b'


    def generate_response(self, prompt: str, image: str, jsonDescription: BaseModel) -> list:
        if not image:
            raise ValueError("Image is required for classification.")
        
        if not prompt:
            raise ValueError("Prompt is required for classification.")

        try:
            # Set up chat as usual
            response = chat(
                model= self.model,
                format=jsonDescription.model_json_schema(),  # Pass in the schema for the response
                messages=[
                    {
                        'role': 'system',
                        'content': f"You are an image classifier. Use the ImageNet categories to classify images. return as JSON"
                    },
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image],
                    },
                ],
                options={'temperature': 0},  # Set temperature to 0 for more deterministic output
            )

            image_analysis = jsonDescription.model_validate_json(response.message.content)
            
            return image_analysis
        except TypeError as e:
            raise RuntimeError(f"Error in ollama.chat: {e}")
