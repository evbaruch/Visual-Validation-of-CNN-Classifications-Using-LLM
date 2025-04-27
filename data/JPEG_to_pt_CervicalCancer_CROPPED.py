import os
import torch
from PIL import Image
import CCDataSet_init as CCD
from tqdm import tqdm
import global_data as gd
import random  # Import random for sampling

load_image_dir = "data\\source\\CervicalCancer\\JPEG"
save_image_dir = "data\\source\\CervicalCancer\\pt\\COMPLETE"

def process_images(image_dir, imagenet_classes, transform, num_for_categorie):
    """
    Processes images from a directory, converting them to tensors and mapping them to class labels.

    Args:
        image_dir (str): Path to the directory containing image files.
        imagenet_classes (list): List of ImageNet class names.
        transform (callable): Transformation function to preprocess the images.

    Returns:
        tuple: A tuple containing two lists:
            - image_tensors: List of image tensors after transformation.
            - labels: List of class indices corresponding to each image.
    """
    image_tensors = []
    labels = []

    for category in imagenet_classes:
        for type in ["COMPLETE"]:
            path = os.path.join(image_dir, category, type)
            try:
                # Get all filenames in the directory
                all_filenames = os.listdir(path)
                
                # Randomly select `num_for_categorie` files
                selected_filenames = random.sample(all_filenames, min(num_for_categorie, len(all_filenames)))
                
                for filename in selected_filenames:
                    filepath = os.path.join(path, filename)
                    
                    # Open the image and convert to RGB format
                    image = Image.open(filepath).convert("RGB")
                    
                    # Apply the transformation
                    image_tensor = transform(image)
                    image_tensors.append(image_tensor)
                    
                    # Extract the category from the filename
                    labels.append(category)
            except Exception as e:
                print(f"Error processing files in {path}: {e}")
    
    return image_tensors, labels

def main(image_dir, save_image_dir, num_for_categorie):
    """
    Main function to process images in a directory and save the resulting tensors to files.

    Args:
        image_dir (str): Path to the directory containing input images.
        save_image_dir (str): Path to the directory where output tensors will be saved.

    Saves:
        - x_batch.pt: Tensor containing the transformed images.
        - y_batch.pt: Tensor containing the encoded labels.
    """
    # Load ImageNet class names and initialize the transformation function
    imagenet_classes = CCD.classes
    transform = gd.initialize_transform()
    
    # Process images and get their tensors and labels
    image_tensors, labels = process_images(image_dir, imagenet_classes, transform, num_for_categorie)

    # Encode the labels as integers
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in labels]
    
    # Stack image tensors into a single tensor
    x_batch = torch.stack(image_tensors)
    y_batch = torch.tensor(encoded_labels)

    # Ensure the output directory exists
    os.makedirs(f"{save_image_dir}_{num_for_categorie}", exist_ok=True)

    # Save the tensors to files
    torch.save(x_batch, os.path.join(f"{save_image_dir}_{num_for_categorie}", "x_batch.pt"))
    torch.save(y_batch, os.path.join(f"{save_image_dir}_{num_for_categorie}", "y_batch.pt"))

# Run the script if executed directly
if __name__ == "__main__":
    main(load_image_dir, save_image_dir, 40)
