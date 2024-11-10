from openai import OpenAI
import base64
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client with the provided API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Function to encode the image to a base64 string
def encode_image(image_path):
    """
    Encodes an image to a base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to the image file
image_path = "images/Dog1.jpg"

# Getting the base64 encoded string of the image
base64_image = encode_image(image_path)

# Base64 encoding is used to convert binary data (like an image) into a text string.
# This is necessary because the API expects the image data to be in a text format.
# Base64 ensures that the image data can be safely transmitted as part of a JSON payload.

# Create a completion request to the OpenAI API
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{base64_image}",
                    }
                },
            ],
        }
    ],
)

# Print the response from the API
print(completion.choices[0].message)