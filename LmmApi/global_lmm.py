import pandas as pd
import os
import re
import spacy
import os
import string
from collections import defaultdict
from tqdm import tqdm
import re
import numpy as np


# Load the spaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Define a function to clean the text by separating punctuation and removing extra spaces
def clean_text(input_text):
    """
    Cleans text by separating punctuation and removing extra spaces.

    Args:
        input_text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    cleaned_text = re.sub(r'([^\w\s])', r' \1 ', input_text)  # Add spaces around punctuation
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    return cleaned_text.strip()

# Define a function to lemmatize the text using spaCy
def lemmatize_text(input_text):
    """
    Lemmatizes the input text using spaCy after converting it to lowercase.

    Args:
        text (str): The text to lemmatize.

    Returns:
        str: The lemmatized version of the text.
    """
    text = input_text.lower()  # Convert text to lowercase
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Define a function to filter the text by removing stop words, numbers, punctuation, and non-alphabetic characters
def filter_text(input_text):
    """
    Filters the word list by removing stop words, numbers, punctuation,
    and non-alphabetic characters.

    Parameters:
        word_dict (dict): Dictionary of words and their frequencies.

    Returns:
        list: Filtered list of words.
    """
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 
        'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
        'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
        'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
        's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
        'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
        'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
        "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


    filtered = ''
    for word in input_text.split():

        if word.lower() in stop_words:
            continue
        if any(char.isdigit() for char in word):
            continue
        if any(char in string.punctuation for char in word):
            continue
        if not bool(re.fullmatch(r'[A-Za-z]+', word)):
            continue


        filtered += word + ' '

    return filtered