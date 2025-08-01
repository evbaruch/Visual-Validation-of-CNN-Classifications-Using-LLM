import os
import re
from PIL import Image
import pandas as pd
from tqdm import tqdm


# def foo(path: str, save_csv: bool = True):
    
#     xai = "guided_gradcam"
#     P = "05"
#     columns = ['Index', 'True_Label', 'Match', 'llm_1']
    
#     for image in os.listdir(f"{path}/{xai}/{xai}{P}"):
#         for P in ["05","10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95"]:
#             for xai in ["guided_backprop","guided_gradcam", "integrated_gradients","inputxgradient", "smoothgrad"]:
#                 print(f"{path}/{xai}/{xai}_{P}/{image}")
#                 file_path = f"{path}/{xai}/{xai}_{P}/{image}"
#                 basename = os.path.basename(file_path)
#                 image_name = basename.split('_', 1)[1].split('.')[0]  # Extracts 'tench'
#                 insex = basename.split('_')[0]  # Extracts the index from the file name
#                 image_name = image_name.replace('_', ' ')  # Replace underscores with spaces for better readability
#                 print(f"Image name: {image_name}")
                
#                 data = [insex, image_name, "correctly"] + ["response"]
                
#                 # add data to xai csv in a new line
#                 csv_path = f"{save_csv}/{xai}/{xai}_{P}_resnet18.csv"
#                 os.makedirs(os.path.dirname(f"{save_csv}/{xai}"), exist_ok=True)
#                 if not os.path.exists(csv_path):
#                     with open(csv_path, 'w') as f:
#                         f.write(','.join(data) + '\n')
                

            
# foo("data/midAvi_grey","data/llm_answer_Avi_grey")

import os

def strip_resnet_suffix(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if "_resnet18" in fname:
                new_name = fname.replace("_resnet18", "")
                src = os.path.join(root, fname)
                dst = os.path.join(root, new_name)
                if os.path.exists(dst):
                    print(f"Skipping {src}; target exists: {dst}")
                    continue
                os.rename(src, dst)
                print(f"Renamed: {src} -> {dst}")

# Example usage:
strip_resnet_suffix("data\llm_answer_Avi_grey")

