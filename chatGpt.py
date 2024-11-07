from openai import OpenAI
import base64

client = OpenAI(api_key = "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")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


image_path = "images/Dog1.jpg"
# Getting the base64 string
base64_image = encode_image(image_path)

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

print(completion.choices[0].message)