
# import  the past code from the folder "past" and the necessary libraries
import sys
from global_val import *
import torch
import os

import past.image_creater as image_creater
import past.classes as classes

import pandas as pd

def select_and_save_top_samples(samples: int, top_k: int, save_path = DATA_PATH , data_path = IMAGENET_PATH):
    
    # Load device: Check if a CUDA-compatible GPU is available and set the computation device accordingly.
    # If a GPU is available, it sets the device to 'cuda'; otherwise, it defaults to 'cpu'.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
        
    # TODO: Rename to clearer names (meaningful names)
    x = os.path.join(data_path, X)
    y = os.path.join(data_path, Y)
    s = os.path.join(data_path, S)
        
    # Load image batches: Loads `nr_samples` samples for x (input images), y (labels), and s (saliency maps)
    # from the specified links and moves them to the given device (CPU or GPU).
    x_batch, y_batch, s_batch = image_creater.load_images(samples, x, y, s, device)
    















































# def filter_with_model(threshold: float, method: str, pre_trained_model: function):
#     pass
