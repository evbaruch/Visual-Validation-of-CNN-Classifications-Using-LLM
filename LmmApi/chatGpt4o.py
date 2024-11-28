import base64
import os
from dotenv import load_dotenv
from openai import OpenAI
from LmmApi.LLMStrategy import LLMStrategy

class ChatGPT4O(LLMStrategy): # ChatGPT4O class implements the LLMStrategy interface
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encodes an image to a base64 string.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The base64 encoded string of the image.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_response(self, prompt: str, image: str) -> str:
        """
        Generates a response using OpenAI's chat completion.

        Args:
            prompt (str): The text prompt to accompany the image.
            image_path (str): The path to the image file.

        Returns:
            str: The generated response from OpenAI.
        """
        # Encode the image to base64
        base64_image = self.encode_image(image)

        # Send the request to OpenAI
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        },
                    ],
                }
            ],
        )

        # Return the API response
        return completion.choices[0].message.content
