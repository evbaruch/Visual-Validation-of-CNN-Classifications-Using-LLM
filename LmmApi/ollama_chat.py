import ollama

model1 = 'llama3.1:8b'
model2 = 'llama3.2-vision:90b'

response = ollama.chat(model='llama3.2-vision:11b', messages=[
  { 
    'role': 'user',
    'content': 'Give me a five-category softmax score based on what you see in the image, And dont add unnecessary text.',
    'images': ['images\Dog2.jpg']
  },
])

print(response['message']['content'])