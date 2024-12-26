
import os

from data import global_data as gd

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import pairwise_distances_argmin

def find_closest_category(raw_label, imagenet_classes):
    """
    Find the most similar category and index from the ImageNet class labels.

    Args:
        raw_label (str): The raw label to match against ImageNet classes.
        imagenet_classes (list): A list of ImageNet class labels.

    Returns:
        int: The index of the closest matching category.
        str: The name of the closest matching category.
    """
    vectorizer = CountVectorizer().fit(imagenet_classes)
    category_vectors = vectorizer.transform(imagenet_classes)
    raw_vector = vectorizer.transform([raw_label])
    closest_idx = pairwise_distances_argmin(raw_vector, category_vectors)[0]

    # Save vectorizer, category_vectors, and raw_vector to CSV files
    vectorizer_df = pd.DataFrame(vectorizer.get_feature_names_out(), columns=['feature'])
    vectorizer_df.to_csv('vectorizer.csv', index=False)

    category_vectors_df = pd.DataFrame(category_vectors.toarray(), columns=vectorizer.get_feature_names_out())
    category_vectors_df.to_csv('category_vectors.csv', index=False)

    raw_vector_df = pd.DataFrame(raw_vector.toarray(), columns=vectorizer.get_feature_names_out())
    raw_vector_df.to_csv('raw_vector.csv', index=False)

    return closest_idx, imagenet_classes[closest_idx]

imagenet_classesA = gd.load_imagenet_classes()
imagenet_classesB = ["Great white shark", "tiger", "shark", "   hammerhead   "]

i = 0
for imeg in imagenet_classesB:
    i = i + 1
    idx, closest = find_closest_category(imeg, imagenet_classesA)
    print(f"{i}: {imeg} --> {idx}: {closest}")


print("Done.")
