from abc import ABC, abstractmethod
import os
import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms



class ImageProcessorABC(ABC):
    def __init__(self, image_dir: str = '', pt_dir: str = ''):
        self.image_dir = image_dir
        self.pt_dir = pt_dir
        self.labels_decoded = {}

    @abstractmethod
    def process_labels(self) -> list:
        """
        Abstract method to load data from the source.
        Subclasses must implement this method.
        """
        pass

    @abstractmethod
    def process_images(self) -> list:
        """
        Abstract method to process images.
        Subclasses must implement this method.
        """
        pass

    def save_tensors(self, images_path: list = None, labels: list = None):
        """
        Saves image tensors and labels to files.
        """
        # Ensure images and labels are processed if not provided
        if not images_path:
            images_path = self.process_images()
        if not labels:
            labels = self.process_labels()
            
        if not os.path.exists(self.pt_dir):
            os.makedirs(self.pt_dir)

        image_tensors = []
        labels_idx = []
        label_to_index = {}  # Dictionary to map labels to indices
        next_index = 0

        for image, label in tqdm(zip(images_path, labels), desc="Saving tensors"):
            # Open the image and convert to RGB format
            image = Image.open(image).convert("RGB")
            
            # Apply the transformation to the image
            image_tensor = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize images to 224x224
                transforms.ToTensor(),          # Convert images to tensors
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
                ])(image)
            
            # Save the image tensor
            image_tensors.append(image_tensor)
            
            # Assign a consistent index to the label
            if label not in label_to_index:
                label_to_index[label] = next_index
                next_index += 1
            labels_idx.append(label_to_index[label])
            
        # Store the label mapping in self.labels_decoded
        self.labels_decoded = {v: k for k, v in label_to_index.items()}
    
            
        # Stack image tensors into a single tensor
        x_batch = torch.stack(image_tensors)
        y_batch = torch.tensor(labels_idx)
        
        # Save the tensors to files
        torch.save(x_batch, os.path.join(self.pt_dir, "x_batch.pt"))
        torch.save(y_batch, os.path.join(self.pt_dir, "y_batch.pt"))
