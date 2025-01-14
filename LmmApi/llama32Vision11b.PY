from LmmApi.LLMStrategy import LLMStrategy
from ollama import chat
from pydantic import BaseModel

# Concrete Strategy: OpenAI GPT 
class llama32Vision11b(LLMStrategy):
    """
    A concrete implementation of the LLMStrategy class that utilizes the Llama 3.2 vision model for image classification.
    This class communicates with an image classification API (ollama.chat) to classify images using ImageNet categories.
    """
    def __init__(self):
        """
        Initializes the llama32Vision11b class with the model identifier for Llama 3.2 vision model (11b).
        """
        self.model = 'llama3.2-vision:11b'


    def generate_response(self, prompt: str, image: str, jsonDescription: BaseModel  | None = None) -> list:
        """
        Generates a classification response based on the provided prompt and image.

        Parameters:
        - prompt (str): The text prompt used for image classification.
        - image (str): The image to be classified (as a file path or URL).
        - jsonDescription (BaseModel | None, optional): An optional Pydantic model to format the response. Defaults to None.

        Returns:
        - list: A list of classified data based on the image and prompt.

        Raises:
        - ValueError: If either the image or prompt is not provided.
        - RuntimeError: If an error occurs during the classification process.
        """
        if not image:
            raise ValueError("Image is required for classification.")
        
        if not prompt:
            raise ValueError("Prompt is required for classification.")
        
        if jsonDescription:
            format = jsonDescription.model_json_schema()
        else:
            format = ""

        try:
            # Set up chat as usual
            response = chat(
                model= self.model,
                format=format,  # Pass in the schema for the response
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
