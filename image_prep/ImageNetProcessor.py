from PIL import Image
import os
from data.ImageProcessorABC import ImageProcessorABC
import requests
from tqdm import tqdm
import json  


class ImageNetProcessor(ImageProcessorABC):
    def __init__(
        self,
        image_dir: str = 'data/source/imagenet_sample/JPEG',
        pt_dir: str = 'data/source/imagenet_sample/pt',
        ):
        """
        Initializes the ImageNetProcessor with directories for loading and saving images.
        
        Args:
            image_dir (str): Directory containing the ImageNet images.
            pt_dir (str): Directory where processed tensors will be saved.
            labels (list): List of labels to be used for processing.
        """
        super().__init__(image_dir, pt_dir)
        self.imagenet_classes_file = "data/source/imagenet_sample/imagenet_classes.txt"
        self.imagenet_classes_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    
    def set_imagenet_classes(self, imagenet_classes_file: str, imagenet_classes_url: str):
        """
        Sets the ImageNet classes file and URL.
        
        Args:
            imagenet_classes_file (str): Path to the ImageNet classes file.
            imagenet_classes_url (str): URL to download the ImageNet classes file if it doesn't exist.
        """
        self.imagenet_classes_file = imagenet_classes_file
        self.imagenet_classes_url = imagenet_classes_url

    def process_labels(self) -> list:
        if not os.path.exists(self.imagenet_classes_file) and self.imagenet_classes_url:
            print(f"File {self.imagenet_classes_file} not found. Downloading...")
            #download_imagenet_classes(self.imagenet_classes_url, self.imagenet_classes_file)
            response = requests.get(self.imagenet_classes_url)
            if response.status_code == 200:
                with open(self.imagenet_classes_file, "w") as f:
                    f.write(response.text)
            else:
                print("Failed to download the ImageNet labels file.")
        try:
            with open(self.imagenet_classes_file, "r") as f:
                # Parse the JSON data
                labels = json.load(f)
                # Format the labels for printing
                formatted_labels = "\n".join(labels)
                return labels
        except Exception as e:
            print(f"Failed to load {self.imagenet_classes_file}: {e}")
    


    def process_images(self) -> list:
        """
        Returns a list of all image file paths in the load_image_dir directory.

        Returns:
            list: A list of image file paths.
        """
        images = []
        try:
            # Walk through the directory and collect all image file paths
            for root, _, files in os.walk(self.image_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):  # Supported image formats
                        images.append(os.path.join(root, file))
            print(f"Found {len(images)} images in {self.image_dir}.")
        except Exception as e:
            print(f"Error while processing images: {e}")
        return images
    
if __name__ == "__main__":
    processor = ImageNetProcessor()
    imagenet_classes = processor.process_labels()
    # print(f"Loaded {len(imagenet_classes)} ImageNet classes.")
    
    # Example usage of process_images
    images = processor.process_images()
    # print(f"Processed {len(images)} images with corresponding labels.")
    
    processor.save_tensors(images_path=images, labels=imagenet_classes)
    
