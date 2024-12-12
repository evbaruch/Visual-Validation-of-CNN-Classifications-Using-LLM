import requests
import torch
import torchvision
from typing import Union
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from IPython.display import clear_output
from IPython.display import Audio
from IPython.core.display import display
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import quantus
import os
from classes import CLASSES


clear_output()

"""## Models"""

# Returns a list of model names that can be used in the code.
def exist_models():
    return ["resnet18()", "v3_small()", "v3_large()", "v3_inception()"]

# Loads a pre-trained ResNet-18 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def resnet18():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained MobileNetV3-Small model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_small():
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained MobileNetV3-Large model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_large():
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained InceptionV3 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_inception():
    model = torchvision.models.inception_v3(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

"""## Images"""

# Load images from a .npy file, extracting x_batch, y_batch, and s_batch for the specified number of images.
def load_images(num, link):
    assets = np.load(link, allow_pickle=True).item()
    x_batch = assets["x_batch"][:num]
    y_batch = assets["y_batch"][:num]
    s_batch = assets["s_batch"].reshape(-1, 1, 224, 224)[:num]
    return x_batch, y_batch, s_batch

# Load images from separate .pt files for x_batch, y_batch, and s_batch, and move them to the specified device.
def load_images(num, x_link, y_link, s_link, device):
    x_batch = torch.load(x_link).to(device)[:num]
    y_batch = torch.load(y_link).to(device)[:num]
    s_batch = torch.load(s_link).reshape(-1, 1, 224, 224).to(device)[:num]
    return x_batch, y_batch, s_batch

"""## Utils"""

# Denormalize an image from the ImageNet format and convert it to a format suitable for display.
def normalize_image(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone().cpu().numpy()
    else:
        arr_copy = arr.copy()

    arr_copy = (np.array(arr_copy) * std.reshape(-1, 1, 1)) + mean.reshape(-1, 1, 1)
    arr_copy  = np.moveaxis(arr_copy, 0, -1)
    arr_copy = (arr_copy * 255.).astype(np.uint8)
    return arr_copy

# Mask pixels in x_batch based on the threshold applied to a_batch, setting pixels below the threshold to 1.
def remove_pixels(a_batch, x_batch, threshold):
  a_copied_matrix = np.copy(a_batch)
  a_copied_matrix[a_copied_matrix <= threshold] = 0  # Set pixels below threshold to 0
  a_mask = (a_copied_matrix != 0)
  a_reshaped_mask = np.repeat(a_mask, x_batch.shape[1], axis=1) ## מיותר???
  a_masked_x_batch = np.copy(x_batch)
  a_masked_x_batch[~a_reshaped_mask] = 1

  total_pixels = np.prod(x_batch.shape)
  removed_pixels = np.sum(~a_reshaped_mask)

  return a_masked_x_batch, removed_pixels / total_pixels

# Improved version of `remove_pixels`, compatible with both torch tensors and numpy arrays.
def new_remove_pixels(a_batch, x_batch, threshold):
    # Ensure a_batch and x_batch are both NumPy arrays for compatibility with NumPy operations
    if isinstance(a_batch, torch.Tensor):
        a_batch = a_batch.cpu().numpy()
    if isinstance(x_batch, torch.Tensor):
        x_batch_cpu = x_batch.cpu().numpy()
    else:
        x_batch_cpu = x_batch  # Already a NumPy array

    # Copy and apply threshold
    a_copied_matrix = np.copy(a_batch)
    a_copied_matrix[a_copied_matrix <= threshold] = 0  # Set pixels below threshold to 0
    a_mask = (a_copied_matrix != 0)
    
    # Repeat mask along the appropriate axis
    a_reshaped_mask = np.repeat(a_mask, x_batch_cpu.shape[1], axis=1)
    a_masked_x_batch = np.copy(x_batch_cpu)
    a_masked_x_batch[~a_reshaped_mask] = 1

    # Calculate total and removed pixels
    total_pixels = np.prod(x_batch_cpu.shape)
    removed_pixels = np.sum(~a_reshaped_mask)

    # Convert back to torch tensor if needed
    if isinstance(x_batch, torch.Tensor):
        a_masked_x_batch = torch.from_numpy(a_masked_x_batch).to(x_batch.device)

    return a_masked_x_batch, removed_pixels / total_pixels

# Load and play an audio file from a specified link.
def beep(link):
    audio_file = link # Load the audio file and get the audio data and sampling rate
    audio_data, sampling_rate = librosa.load(audio_file, sr=None)
    
    #TODO: save the audio files to a folder
    os.makedirs("audio", exist_ok=True)
    librosa.output.write_wav("audio/beep.wav" + link, audio_data, sampling_rate)

"""## Probabilities"""

# Get probability distributions over classes for a batch of images using a given model.
def get_probabilities(batch, model):
  probabilities = [0] * len(batch)

  for i in range(len(batch)):
    input_image = Image.fromarray(normalize_image(batch[i]))  # Assuming the pixel values are in the range [0, 1]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch) #
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities[i] = torch.nn.functional.softmax(output[0], dim=0)
  return probabilities

"""## Results"""

# Generate a DataFrame with the top 5 predictions and probabilities for each image in the batch.
def get_results5(batch, y_batch, model):
  probabilities = get_probabilities(batch, model)

  results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                     'Class_1', 'Probability_1',
                                     'Class_2', 'Probability_2',
                                     'Class_3', 'Probability_3',
                                     'Class_4', 'Probability_4',
                                     'Class_5', 'Probability_5',])

  for i in range(len(batch)):
      top5_prob, top5_catid = torch.topk(probabilities[i], 5)
      class_list = [0] * top5_prob.size(0)
      prob_list = [0] * top5_prob.size(0)
      for j in range(top5_prob.size(0)):
        class_list[j] = categories[top5_catid[j]]
        prob_list[j] = top5_prob[j].item()

      results_df = results_df._append({'Image': i + 1,
                                      'Real': y_batch[i],
                                      'Name': CLASSES[y_batch[i]],
                                      'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                      'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                      'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                      'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                      'Class_5': class_list[4], 'Probability_5': prob_list[4]}, ignore_index=True)
  clear_output()
  return results_df

# Improved version of `get_results5` that handles both CUDA tensors and missing labels in CLASSES.
def new_get_results5(batch, y_batch, model):
    probabilities = get_probabilities(batch, model)

    results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                       'Class_1', 'Probability_1',
                                       'Class_2', 'Probability_2',
                                       'Class_3', 'Probability_3',
                                       'Class_4', 'Probability_4',
                                       'Class_5', 'Probability_5'])

    for i in range(len(batch)):
        # Get top 5 probabilities and category IDs
        top5_prob, top5_catid = torch.topk(probabilities[i], 5)
        class_list = [0] * top5_prob.size(0)
        prob_list = [0] * top5_prob.size(0)

        for j in range(top5_prob.size(0)):
            class_list[j] = categories[top5_catid[j]]
            prob_list[j] = top5_prob[j].item()

        # Convert y_batch[i] to an integer label if it's still a CUDA tensor
        real_label = int(y_batch[i].cpu().item()) if isinstance(y_batch[i], torch.Tensor) else y_batch[i]

        # Check if real_label exists in CLASSES
        if real_label in CLASSES:
            real_name = CLASSES[real_label]
        else:
            real_name = "Unknown"  # Fallback if label not found in CLASSES

        # Append row to the results DataFrame
        results_df = results_df._append({'Image': i + 1,
                                         'Real': real_label,
                                         'Name': real_name,
                                         'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                         'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                         'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                         'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                         'Class_5': class_list[4], 'Probability_5': prob_list[4]}, 
                                         ignore_index=True)

    clear_output()
    return results_df

# Generate a DataFrame with the top 1 prediction and probability for each image in the batch.
def get_results1(batch, y_batch, model):
  probabilities = get_probabilities(batch, model)

  results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                     'Class_1', 'Probability_1'])

  for i in range(len(batch)):
      top1_prob, top1_catid = torch.topk(probabilities[i], 1)
      class_list = [0] * top1_prob.size(0)
      prob_list = [0] * top1_prob.size(0)
      for j in range(top1_prob.size(0)):
        class_list[j] = categories[top1_catid[j]]
        prob_list[j] = top1_prob[j].item()

      results_df = results_df._append({'Image': i + 1,
                                      'Real': y_batch[i],
                                      'Name': CLASSES[y_batch[i]],
                                      'Class_1': class_list[0], 'Probability_1': prob_list[0]}, ignore_index=True)
  clear_output()
  return results_df

"""## Accuracies"""

# Checks if the actual label of an image is within the top-k predictions for that image.
def is_classified_correct(row, k):
    name_labels = [label.strip() for label in row['Name'].split(',')]

    for i in range(1, k + 1):
        class_label = row[f'Class_{i}']
        if any(label in class_label for label in name_labels):
            return True
    return False

# Calculates and displays the proportion of correctly classified images based on top-k accuracy.
def get_corrects(df, k):
  df['Match'] = df.apply(is_classified_correct, k=k, axis=1)
  print(f"Correctly classified instances: {(df['Match'].sum() / len(df))}")

"""## Main"""

# Generates classification results for the batch and retrieves top-k predictions accuracy.
def get_results(model, a_masked_x_batch, y_batch, k):
  if k == 1:
    df = get_results1(a_masked_x_batch, y_batch, model)
  elif k == 5:
    df = new_get_results5(a_masked_x_batch, y_batch, model)
  else:
    print("Error")
    return 0
  return df

"""## Heatmap"""

def create_heatmap(a_matrix, threshold, image_index):
    # Move to CPU and convert to NumPy if a_matrix is a tensor
    if isinstance(a_matrix, torch.Tensor):
        a_matrix = a_matrix.cpu().numpy()
    
    # Reduce 3D to 2D by averaging across the color channels
    if a_matrix.ndim == 3:
        a_matrix = a_matrix.mean(axis=0)

    # Normalize the importance matrix and apply threshold if needed
    plt.figure(figsize=(8, 8))
    sns.heatmap(a_matrix, cmap='viridis', cbar=True)

    # TODO: save all the images to a folder 
    os.makedirs("images2", exist_ok=True)
    plt.savefig(f"images2/heatmap_{image_index}.png")
    



"""## Catagories"""

# Define the URL where the ImageNet labels are stored
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Download the ImageNet labels file from the URL and save it locally
response = requests.get(url)
if response.status_code == 200:
    with open("imagenet_classes.txt", "w") as f:
        f.write(response.text) 
else:
    print("Failed to download the ImageNet labels file.") 

# Open the downloaded file and read the categories line by line
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]


# Update the paths to your local file locations
x_link = "imagenet_samples/x_batch.pt"
y_link = "imagenet_samples/y_batch.pt"
s_link = "imagenet_samples/s_batch.pt"
SAVE_FILES_LINK = "data/"
BEEP_LINK = "data/beep-01a.mp3"

def main():
    print("main")

    # Load device: Check if a CUDA-compatible GPU is available and set the computation device accordingly.
    # If a GPU is available, it sets the device to 'cuda'; otherwise, it defaults to 'cpu'.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Set the number of samples to process
    nr_samples = 200

    # Load image batches: Loads `nr_samples` samples for x (input images), y (labels), and s (saliency maps)
    # from the specified links and moves them to the given device (CPU or GPU).
    x_batch, y_batch, s_batch = load_images(nr_samples, x_link, y_link, s_link, device)

    # Print the number of matches (samples) loaded.
    print(f"{len(x_batch)} matches found\n")

    # Set the number of top-k predictions to consider for accuracy evaluation
    k = 5

    # Set the threshold for pixel removal in the saliency mask
    threshold = 0.01

    # Define the explanation method to be used
    method = 'Saliency'

    # Load the pre-trained model `v3_small` and assign it to the device for computation
    model, device = v3_small()

    # Generate explanations: Uses the `quantus.explain` function with the selected method
    # to calculate saliency maps (`a_batch`) based on the model’s predictions for `x_batch`.
    a_batch = quantus.explain(model, x_batch, y_batch, method=method, device=device)

    # Remove pixels below the specified threshold in the explanation maps and calculate the masked x_batch.
    # `a_masked_x_batch` is the result of applying the mask, and `removed` gives the proportion of pixels removed.
    a_masked_x_batch, removed = new_remove_pixels(a_batch, x_batch, threshold)

    # Retrieve results and accuracy: Evaluate the top-k predictions based on the masked batch.
    # The function `get_results` returns a DataFrame with results.
    df = get_results(model, a_masked_x_batch, y_batch, k)  # updated v2.0

    # Print the proportion of area removed from the instances (masked pixels).
    print("Area removed from instances: " + str(removed))

    # Calculate and print classification accuracy based on the top-k matches in `df`.
    get_corrects(df, k)

    # Save the results DataFrame as a CSV file with a filename that includes the method, threshold, and k-value.
    df.to_csv(f"{SAVE_FILES_LINK}a_batch_{method}_{threshold}_{k}.csv", index=False)

    # Play a sound from the specified file link to signal completion of processing.
    #beep(BEEP_LINK)

    # Normalize the first image from the masked batch using the normalization function for saliency maps.
    normalize_image(a_masked_x_batch[0])

    # Create a dictionary mapping explanation method names to their corresponding saliency maps.
    explanations = {
        "Saliency": a_batch,
    }

    # Set the index of the image to be displayed
    index = 11

    # Create a subplot with 1 + len(explanations) columns (1 for the original image, others for explanations)
    fig, axes = plt.subplots(nrows=1, ncols=1+len(explanations), figsize=(15, 8))

    # Display the original image, denormalized, with normalization applied to the ImageNet standard.
    # The image is denormalized using the mean and std values for ImageNet, and the axis is turned off.
    axes[0].imshow(np.moveaxis(quantus.normalise_func.denormalise(x_batch[index].cpu().numpy(),
                                                              mean=np.array([0.485, 0.456, 0.406]),
                                                              std=np.array([0.229, 0.224, 0.225])), 0, -1), vmin=0.0, vmax=1.0)

    # Set the title of the first subplot (original image) to show its corresponding ImageNet class.
    axes[0].title.set_text(f"ImageNet class {y_batch[index].item()}")

    # Hide the axes of the original image
    axes[0].axis("off")

    # Loop through the explanations dictionary and display each saliency map in the subsequent subplots.
    # The saliency maps are normalized and displayed using the 'seismic' color map.
    for i, (k, v) in enumerate(explanations.items()):
        axes[i+1].imshow(quantus.normalise_func.normalise_by_negative(explanations[k][index].reshape(224, 224)), cmap="seismic", vmin=-1.0, vmax=1.0)
        axes[i+1].title.set_text(f"{k}")  # Set the title of each subplot to the explanation method name
        axes[i+1].axis("off")  # Hide the axes for the explanation maps

    # Loop through each image in the batch (`a_masked_x_batch`) by iterating over its first dimension (batch size).
    for i in range(a_masked_x_batch.shape[0]):
        # Generate and display a heatmap for the i-th image in the batch using the `create_heatmap()` function.
       create_heatmap(a_masked_x_batch[i], threshold, i)

    """## Info"""

    # Call the `exist_models()` function to check or load any existing models.
    exist_models()

    # Call the `available_methods_captum()` function from the `quantus.helpers.constants` module to list available explanation methods in Captum.
    quantus.helpers.constants.available_methods_captum()

    # Call the `available_metrics()` function from the `quantus.helpers.constants` module to list available metrics for evaluation.
    quantus.helpers.constants.available_metrics()

    """## Images"""

    # Normalize the second image from the masked batch (`a_masked_x_batch`) using the `normalize_image()` function.
    normalize_image(a_masked_x_batch[1])

    # Normalize the second image from the saliency batch (`a_batch`) using the `normalize_image()` function.
    normalize_image(a_batch[1])


if __name__ == "__main__":
    main()