import torch
from dataset_API import image_creater as imc
import os
import quantus
from data import global_data as gd
from PIL import Image


class dataset_interface:

    def __init__(self, data_path: str, save_path: str, samples: int = 200):

        self.data_path = data_path
        self.save_path = save_path
        self.samples = samples

        self.top_k = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.removed = []
        self.Correctly = []


    def filter_with_model(self, threshold: float, method: str, pre_trained_model: str):
        """
        Filters the dataset using a pre-trained model and an explanation method.

        Args:
            threshold (float): The threshold value for removing pixels.
            method (str): The explanation method to use.
            pre_trained_model (str): The pre-trained model to use.

        Returns:
            str: A string containing the method, threshold, pre-trained model name, 
                the proportion of removed pixels, and the classification accuracy.
        """

        # Define the subfolder path for the current method, threshold, and pre-trained model
        subfolder_path = os.path.join(
            self.save_path, f"{method}", f"{method}_{threshold}_{pre_trained_model}"
        )

        # Check if the subfolder already exists
        if os.path.exists(subfolder_path):
            print(f"Skipping processing for {subfolder_path} as it already exists.")
            return f"Skipped {method} {threshold} {pre_trained_model}"

        # Proceed with processing if the subfolder does not exist
        x_link = os.path.join(self.data_path, "x_batch.pt")
        y_link = os.path.join(self.data_path, "y_batch.pt")

        # Load image batches
        x_batch, y_batch = imc.load_images(self.samples, x_link, y_link, self.device)

        # Load the pre-trained model
        exist_models = imc.exist_models()
        if pre_trained_model == exist_models[0]:
            model = imc.resnet18(self.device)
        elif pre_trained_model == exist_models[1]:
            model = imc.v3_small(self.device)
        elif pre_trained_model == exist_models[2]:
            model = imc.v3_large(self.device)
        elif pre_trained_model == exist_models[3]:
            model = imc.v3_inception(self.device)

        # Generate explanations
        if method == "Random":
            a_batch = quantus.explain(model, x_batch, y_batch, method='Saliency', device=self.device)
            a_masked_x_batch, removed = imc.random_remove_pixels(a_batch, x_batch, threshold)
        else:
            a_batch = quantus.explain(model, x_batch, y_batch, method=method, device=self.device)
            a_masked_x_batch, removed = imc.new_remove_pixels(a_batch, x_batch, threshold)

        self.removed.append(removed)

        categories = gd.load_imagenet_classes()

        # Retrieve results and accuracy
        df = imc.new_get_results5(a_masked_x_batch, y_batch, model, categories)
        Correctly = imc.get_corrects(df, self.top_k)
        self.Correctly.append(Correctly)

        # Save results to CSV
        csv_dir = os.path.join(self.save_path, f"{method}", "csv")
        os.makedirs(csv_dir, exist_ok=True)
        df.to_csv(os.path.join(csv_dir, f"{method}_{threshold}_{pre_trained_model}.csv"), index=False)

        # Save images to the subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        for imeg, label, i in zip(a_masked_x_batch, y_batch, range(len(a_masked_x_batch))):
            img = imc.change_ImageNet_format(imeg)
            img = Image.fromarray(img.astype('uint8'))
            imeg_label_idx = int(label.cpu().item()) if isinstance(label, torch.Tensor) else label
            real_name = categories[imeg_label_idx]
            img.save(os.path.join(subfolder_path, f"{i}_{real_name}.png"))

        return f"{method} {threshold} {pre_trained_model} removed: {removed} Correctly: {Correctly}"    

    @staticmethod
    def parse_file_name(path):
        """
        Parses a file path or file name and extracts three components: 
        method, threshold, and pre-trained model name. 
        The file name must follow the format: 'method_threshold_pre_trained_model'.

        Args:
            path (str): The file path or file name.

        Returns:
            tuple: A tuple containing:
                - method (str): The method name.
                - threshold (str): The threshold value.
                - pre_trained_model (str): The name of the pre-trained model.

        Raises:
            ValueError: If the file name does not have the expected format.
        """
        # Extract the file name without the directory
        file_name = os.path.basename(path)
        
        # Remove the file extension if present
        file_name = os.path.splitext(file_name)[0]
        
        # Split the name into parts based on underscores
        parts = file_name.split('_', 2)
        
        if len(parts) != 3:
            raise ValueError("Invalid file name format. Expected format: 'method_threshold_preTrainedModel'")
        
        # Extract parts
        method, threshold, pre_trained_model = parts
        return method, threshold, pre_trained_model