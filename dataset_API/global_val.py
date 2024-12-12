import requests


IMAGENET_PATH = "imagenet_samples/"
X = "x_batch.pt"
Y = "y_batch.pt"
S = "s_batch.pt"
DATA_PATH = "data/"

BEEP_PATH = "data/beep-01a.mp3"

# Define the URL where the ImageNet labels are stored
URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"


def get_imageNet_data():
    
    # Download the ImageNet labels file from the URL and save it locally
    response = requests.get(URL)
    if response.status_code == 200:
        with open("imagenet_classes.txt", "w") as f:
            f.write(response.text) 
    else:
        print("Failed to download the ImageNet labels file.") 

    # Open the downloaded file and read the categories line by line
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]