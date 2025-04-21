# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
prahladmehandiratta_cervical_cancer_largest_dataset_sipakmed_path = kagglehub.dataset_download('prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed')

print('Data source import complete.')
print(prahladmehandiratta_cervical_cancer_largest_dataset_sipakmed_path)

# Importing Necessary Libraries
import cv2
import os
import shutil
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Function for Formatting Dataset
def FormatDataset(dataset_src, dataset_dest, classes):
    # Making a Copy of Dataset
    new_cropped_dest = [os.path.join(dataset_dest, cls, 'CROPPED') for cls in classes]
    new_complete_dest = [os.path.join(dataset_dest, cls, 'COMPLETE') for cls in classes]
    # Updated to use os.path.join to ensure platform compatibility when building paths.
    cropped_src = [os.path.join(dataset_src, "im_" + cls, "im_" + cls, "CROPPED") for cls in classes ]
    complete_src = [os.path.join(dataset_src, "im_" + cls, "im_" + cls) for cls in classes ]
    for (dest1, dest2) in zip(new_cropped_dest, new_complete_dest):
        os.makedirs(dest1, exist_ok=True); # exist_ok=True to avoid error if directory exists
        os.makedirs(dest2, exist_ok=True)
    # Formating Cropped Images
    for (src,new_dest) in zip(cropped_src, new_cropped_dest):
        # Check if directory exists before trying to list files within
        if os.path.exists(src):
            for file in os.listdir(src):
                filename, file_ext = os.path.splitext(file)
                if file_ext == '.bmp':
                    img_des = os.path.join(new_dest, filename + '.jpg')
                    img = cv2.imread(os.path.join(src, file))
                    img = cv2.resize(img, (64, 64))
                    img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    img = cv2.blur(img, (2, 2))
                    cv2.imwrite(img_des ,img)
    # Formatting Complete Images
    for (src,new_dest) in zip(complete_src, new_complete_dest):
        # Check if directory exists before trying to list files within
        if os.path.exists(src):
            for file in os.listdir(src):
                filename, file_ext = os.path.splitext(file)
                if file_ext == '.bmp':
                    img_des = os.path.join(new_dest, filename + '.jpg')
                    img = cv2.imread(os.path.join(src, file))
                    img = cv2.resize(img, (256, 256))
                    img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                    img = cv2.blur(img, (2, 2))
                    cv2.imwrite(img_des ,img)

# Source Location for Dataset
# Use variable returned by kagglehub.dataset_download
src = prahladmehandiratta_cervical_cancer_largest_dataset_sipakmed_path
# Destination Location for Dataset
dest = 'data'
# Image Classes
classes = ["Dyskeratotic","Koilocytotic","Metaplastic","Parabasal","Superficial-Intermediate"]
# Formatting Dataset
FormatDataset(src, dest, classes)

# root_dir = "./CervicalCancer"
# classes = ["Dyskeratotic","Koilocytotic","Metaplastic","Parabasal","Superficial-Intermediate"]

# def GetDatasetSize(path, classes, main = "CROPPED"):
#     num_of_image = {}
#     for cls in classes:
#         # Counting the Number of Files in the Folder
#         num_of_image[cls] = len(os.listdir(os.path.join(path, cls, main)))
#     return num_of_image

# print(GetDatasetSize(root_dir, classes, "COMPLETE"))