import os
import torch
from PIL import Image
import global_data as gd
from tqdm import tqdm

load_image_dir = "data\\source\\imagenet-sample2\\JPEG"
save_image_dir = "data\\source\\imagenet-sample2\\pt"

def process_images(image_dir, imagenet_classes, transform):
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
    
    for filename in tqdm(os.listdir(image_dir), desc=f"Processing images in {image_dir}"):
        if filename.endswith(".JPEG"):
            filepath = os.path.join(image_dir, filename)
            try:
                # Open the image and convert to RGB format
                image = Image.open(filepath).convert("RGB")
                
                # Apply the transformation
                image_tensor = transform(image)
                image_tensors.append(image_tensor)
                
                # Extract the category from the filename
                raw_label = " ".join(filename.split("_")[1:]).rsplit(".", 1)[0].replace("_", " ")

                # Map the raw label to the closest ImageNet class index
                closest_label = gd.find_closest_category_idx(raw_label, imagenet_classes)
                labels.append(closest_label)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return image_tensors, labels

def main(image_dir, save_image_dir):
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
    imagenet_classes = gd.load_imagenet_classes()
    transform = gd.initialize_transform()
    
    # Process images and get their tensors and labels
    image_tensors, labels = process_images(image_dir, imagenet_classes, transform)

    # Encode the labels as integers
    unique_labels = list(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_to_index[label] for label in labels]
    
    # Stack image tensors into a single tensor
    x_batch = torch.stack(image_tensors)
    y_batch = torch.tensor(encoded_labels)

    # Ensure the output directory exists
    os.makedirs(save_image_dir, exist_ok=True)

    # Save the tensors to files
    torch.save(x_batch, os.path.join(save_image_dir, "x_batch.pt"))
    torch.save(y_batch, os.path.join(save_image_dir, "y_batch.pt"))

# Run the script if executed directly
if __name__ == "__main__":
    main(load_image_dir, save_image_dir)
