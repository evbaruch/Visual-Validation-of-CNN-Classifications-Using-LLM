import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import requests

# URL and default path for ImageNet class labels
imagenet_classes_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes_file = "data\\imagenet_classes.txt"

def load_imagenet_classes(filepath=imagenet_classes_file, url=imagenet_classes_url):
    """
    Load categories from a file, downloading it if it does not exist.
    
    Args:
        filepath (str): Local path to the file containing ImageNet class labels.
        url (str): URL to download the file if it is not found locally.

    Returns:
        list: A list of class labels.

    Raises:
        Exception: If the file cannot be loaded after retrying.
    """
    if not os.path.exists(filepath) and url:
        print(f"File {filepath} not found. Downloading...")
        download_imagenet_classes(url, filepath)
    try:
        with open(filepath, "r") as f:
            return [line.strip() for line in f]
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        if url:
            print("Retrying download...")
            download_imagenet_classes(url, filepath)
            with open(filepath, "r") as f:
                return [line.strip() for line in f]
        else:
            raise

def initialize_transform():
    """
    Initialize the image transformation pipeline.

    Returns:
        torchvision.transforms.Compose: A composed transformation that resizes
        images to 224x224 pixels and converts them to tensors.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])

def find_closest_category(raw_label, imagenet_classes):
    """
    Find the most similar category index from the ImageNet class labels.

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
    return closest_idx, imagenet_classes[closest_idx]



def download_imagenet_classes(url: str, output_file: str):
    """
    Download the ImageNet labels file from the given URL and save it locally.

    Args:
        url (str): URL of the ImageNet class labels file.
        output_file (str): Local path to save the downloaded file.

    Raises:
        Exception: If the download fails (non-200 status code).
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, "w") as f:
            f.write(response.text)
    else:
        print("Failed to download the ImageNet labels file.")