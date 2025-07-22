from PIL import Image
import os
from data.ImageProcessorABC import ImageProcessorABC
import requests
from tqdm import tqdm
import json  


class CervicalCancerProcessor(ImageProcessorABC):
    def __init__(
        self,
        image_dir: str = 'data/source/CervicalCancer/JPEG',
        pt_dir: str = 'data/source/CervicalCancer/pt',
        labels: list = None
    ):
        """
        Initializes the CervicalCancerProcessor with directories for loading and saving images.
        
        Args:
            image_dir (str): Directory containing the Cervical Cancer images.
            pt_dir (str): Directory where processed tensors will be saved.
            labels (list): List of labels to be used for processing.
        """
        super().__init__(image_dir, pt_dir)
        self.labels = labels if labels else ['Superficial-Intermediate', 'Dyskeratotic', 'Metaplastic', 'Parabasal', 'Koilocytotic']