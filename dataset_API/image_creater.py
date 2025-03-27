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
#from IPython.core.display import display
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import quantus
import os
from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights, Inception_V3_Weights



clear_output()

"""## Models"""

# Returns a list of model names that can be used in the code.
def exist_models():
    return ["resnet18", "v3_small", "v3_large", "v3_inception"]

"""
# Loads a pre-trained ResNet-18 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def resnet18(device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained MobileNetV3-Small model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_small(device):
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained MobileNetV3-Large model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_large(device):
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained InceptionV3 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_inception(device):
    model = torchvision.models.inception_v3(pretrained=True)
    model = model.to(device)
    model.eval()
    clear_output()
    return model
"""

# Loads a pre-trained ResNet-18 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def resnet18(device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights='IMAGENET1K_V1')
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained MobileNetV3-Small model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_small(device):
    model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained MobileNetV3-Large model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_large(device):
    model = torchvision.models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    clear_output()
    return model

# Loads a pre-trained InceptionV3 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_inception(device):
    model = torchvision.models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()
    clear_output()
    return model


"""## Images"""

# Load images from a .npy file, extracting x_batch, y_batch, and s_batch for the specified number of images.
def load_images(num, link):
    assets = np.load(link, allow_pickle=True).item()
    x_batch = assets["x_batch"][:num]
    y_batch = assets["y_batch"][:num]
    s_batch = assets["s_batch"].reshape(-1, 1, 224, 224)[:num]
    return x_batch, y_batch, s_batch

# Load images from separate .pt files for x_batch, y_batch, and s_batch, and move them to the specified device.
def load_images(nr_samples, x_link, y_link, device):

    x_batch = torch.load(x_link, weights_only=True).to(device)[:nr_samples]
    y_batch = torch.load(y_link, weights_only=True).to(device)[:nr_samples]
    #s_batch = torch.load(s_link, weights_only=True).reshape(-1, 1, 224, 224).to(device)[:nr_samples]

    return x_batch, y_batch


"""## Utils"""

# Denormalize an image from the ImageNet format and convert it to a format suitable for display.
def change_ImageNet_format(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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

    # Calculate total and removed pixels per image
    removed_pixels_per_image = np.sum(~a_reshaped_mask, axis=(1, 2, 3)).tolist()

    # Convert back to torch tensor if needed
    if isinstance(x_batch, torch.Tensor):
        a_masked_x_batch = torch.from_numpy(a_masked_x_batch).to(x_batch.device)

    return a_masked_x_batch, removed_pixels_per_image

def random_remove_pixels(a_batch, x_batch, threshold):
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
    
    # Randomly shuffle the mask
    a_mask_flat = a_mask.flatten()
    np.random.shuffle(a_mask_flat)
    a_mask_shuffled = a_mask_flat.reshape(a_mask.shape)
    
    # Repeat mask along the appropriate axis
    a_reshaped_mask = np.repeat(a_mask_shuffled, x_batch_cpu.shape[1], axis=1)
    a_masked_x_batch = np.copy(x_batch_cpu)
    a_masked_x_batch[~a_reshaped_mask] = 1

    # Calculate total and removed pixels per image
    removed_pixels_per_image = np.sum(~a_reshaped_mask, axis=(1, 2, 3)).tolist()

    # Convert back to torch tensor if needed
    if isinstance(x_batch, torch.Tensor):
        a_masked_x_batch = torch.from_numpy(a_masked_x_batch).to(x_batch.device)

    return a_masked_x_batch, removed_pixels_per_image

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
    input_image = Image.fromarray(change_ImageNet_format(batch[i]))  # Assuming the pixel values are in the range [0, 1]

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
def get_results5(batch, y_batch, model, categories):
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
                                      'Name': categories[y_batch[i]],
                                      'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                      'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                      'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                      'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                      'Class_5': class_list[4], 'Probability_5': prob_list[4]}, ignore_index=True)
  clear_output()
  return results_df

# Improved version of `get_results5` that handles both CUDA tensors and missing labels in CLASSES.
def new_get_results5(x_batch, y_batch, model, categories, removed_pixels):
    probabilities = get_probabilities(x_batch, model)

    results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                       'Class_1', 'Probability_1',
                                       'Class_2', 'Probability_2',
                                       'Class_3', 'Probability_3',
                                       'Class_4', 'Probability_4',
                                       'Class_5', 'Probability_5',
                                       'Removed_Pixels', 'Removed_Percentage'])

    for i in range(len(x_batch)):
        # Get top 5 probabilities and category IDs
        top5_prob, top5_catid = torch.topk(probabilities[i], 5)
        class_list = [0] * top5_prob.size(0)
        prob_list = [0] * top5_prob.size(0)

        for j in range(top5_prob.size(0)):
            class_list[j] = categories[top5_catid[j]]
            prob_list[j] = top5_prob[j].item()

        # Convert y_batch[i] to an integer label if it's still a CUDA tensor
        real_label = int(y_batch[i].cpu().item()) if isinstance(y_batch[i], torch.Tensor) else y_batch[i]
        real_name = categories[real_label]

        # Append row to the results DataFrame
        results_df = results_df._append({'Image': i + 1,
                                         'Id': real_label,
                                         'Name': real_name,
                                         'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                         'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                         'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                         'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                         'Class_5': class_list[4], 'Probability_5': prob_list[4],
                                         'Removed_Pixels': removed_pixels[i],
                                         'Removed_Percentage': (removed_pixels[i]/(224*224*3))*100 }, 
                                         ignore_index=True)

    clear_output()
    return results_df

# Generate a DataFrame with the top 1 prediction and probability for each image in the batch.
def get_results1(batch, y_batch, model, categories):
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
                                      'Name': categories[y_batch[i]],
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
  return (df['Match'].sum() / len(df))


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

def create_heatmap1(a_matrix, threshold, image_index):
    # Move to CPU and convert to NumPy if a_matrix is a tensor
    if isinstance(a_matrix, torch.Tensor):
        a_matrix = a_matrix.cpu().numpy()
    
    # Reduce 3D to 2D by averaging across the color channels
    if a_matrix.ndim == 3:
        a_matrix = a_matrix.mean(axis=0)

    # Normalize the importance matrix and apply threshold if needed
    plt.figure(figsize=(8, 8))
    sns.heatmap(a_matrix, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Pixel Importance - Image {image_index + 1}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    # TODO: save all the images to a folder 
    os.makedirs("images2", exist_ok=True)
    plt.savefig(f"images2/heatmap_{image_index}.png")
    

def create_heatmap2(a_matrix, threshold, image_index):
    # Move to CPU and convert to NumPy if a_matrix is a tensor
    if isinstance(a_matrix, torch.Tensor):
        a_matrix = a_matrix.cpu().numpy()

    importance_matrix = np.where(a_matrix > threshold, a_matrix, 0)
    importance_matrix = importance_matrix / np.max(importance_matrix)

    plt.figure(figsize=(8, 8))
    sns.heatmap(importance_matrix[0], cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Pixel Importance - Image {image_index + 1}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()
    
    # TODO: save all the images to a folder

"""## Catagories"""

"""
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

# Load the categories from the file
with open("data\\imagenet_classes.txt", "r") as f:
    imagenet_classes = [line.strip() for line in f]


# Update the paths to your local file locations
x_link = "data\\source\\imagenet-sample2\\pt\\x_batch.pt"
y_link = "data\\source\\imagenet-sample2\\pt\\y_batch.pt"
s_link = ""
SAVE_FILES_LINK = "data/"
BEEP_LINK = "data/beep-01a.mp3"
CLASSES = imagenet_classes

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
    x_batch, y_batch = load_images(nr_samples, x_link, y_link, device)

    # Print the number of matches (samples) loaded.
    print(f"{len(x_batch)} x_batch matches found\n")
    print(f"{len(y_batch)} y_batch matches found\n")


    # Set the number of top-k predictions to consider for accuracy evaluation
    k = 5

    # Set the threshold for pixel removal in the saliency mask
    threshold = 0.01

    # Define the explanation method to be used
    method = 'Saliency'

    # Load the pre-trained model `v3_small` and assign it to the device for computation
    model = v3_small(device)

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
    Correctly = get_corrects(df, k)
    print(f"Correctly classified instances: {Correctly}")

    # Save the results DataFrame as a CSV file with a filename that includes the method, threshold, and k-value.
    df.to_csv(f"{SAVE_FILES_LINK}a_batch_{method}_{threshold}_{k}.csv", index=False)

    # Play a sound from the specified file link to signal completion of processing.
    #beep(BEEP_LINK)
    # Directory containing the images



    # Normalize the first image from the masked batch using the normalization function for saliency maps.
    for imeg, label, i in zip(a_masked_x_batch, y_batch, range(len(a_masked_x_batch))):  # Use zip to iterate over both lists simultaneously
        

        # Format the image
        a = change_ImageNet_format(imeg)
        
        # Convert the array to an image
        img = Image.fromarray(a.astype('uint8'))  # Ensure the data is in uint8 format for saving
    
        # Handle label if it's a Tensor
        imeg_label = int(label.cpu().item()) if isinstance(label, torch.Tensor) else label

        real_name = CLASSES[imeg_label]
        print(f"label:  {label} -->  {real_name}")

        # Save the image as PNG
        img.save(f"data\\temp\\{i} {real_name}.png")  # Use forward slash or raw string literal for file paths


if __name__ == "__main__":
    main()

"""