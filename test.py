
import os
from gensim.models import KeyedVectors
from data import global_data as gd
import gensim.downloader as api

model_path = "word2vec-google-news-300.model"

# Check if the model is already saved locally
if not os.path.exists(model_path):
    # Download the model and save it locally
    model = api.load("word2vec-google-news-300")
    model.save(model_path)
else:
    # Load the model from the local file
    model = KeyedVectors.load(model_path)
imagenet_classesA = gd.load_imagenet_classes()
#imagenet_classesB = ["aaa", "bbb", "ccc", "ddd", "eee", "fff", "ggg", "hhh",]

i = 0
for imeg in imagenet_classesA:
    i = i + 1
    #idx, closest = gd.find_closest_category(imeg, imagenet_classesA)
    sentence_list = [word  for word in imeg.split()]
     
    for word in sentence_list:
    #   model[word]

        if word not in model.key_to_index:
            print(f"{imeg} --> {word} : {i}")

print("Done.")