from openai import OpenAI

client = OpenAI(api_key = "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")

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
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    }
                },
            ],
        }
    ],
)

print(completion.choices[0].message)