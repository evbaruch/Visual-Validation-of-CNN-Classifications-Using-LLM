import torch
import torch.nn as nn
from dataset_API import image_creater as imc
import os
import quantus
from data import global_data as gd
from PIL import Image
import numpy as np


class dataset_interface:

    def __init__(self, data_path: str, save_path: str, samples: int = 200):

        self.data_path = data_path
        self.save_path = save_path
        self.samples = samples
        # self.categories = gd.load_imagenet_classes()
        # {'Superficial-Intermediate': 0, 'Metaplastic': 1, 'Koilocytotic': 2, 'Dyskeratotic': 3, 'Parabasal': 4}
        self.categories = ["Superficial-Intermediate","Metaplastic","Koilocytotic","Dyskeratotic","Parabasal"]


        self.top_k = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        self.removed = []
        self.Correctly = []


    def filter_with_model_batch(self, threshold: float, method: str, pre_trained_model: str , precentage_wise: bool = False):
        """
        Filters the dataset using a pre-trained model and an explanation method in smaller batches.

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

         # Define the path to save the explanation maps
        a_batches_file = os.path.join(self.save_path, f"{method}", f"a_batches {method} {pre_trained_model}.npz")

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

        # Process in smaller batches
        batch_size = 32  # Adjust this value based on your system's memory capacity
        num_batches = (len(x_batch) + batch_size - 1) // batch_size

        # Check if the explanation maps already exist
        if os.path.exists(a_batches_file):
            #print(f"Loading precomputed explanation maps from {a_batches_file}")
            a_batches = np.load(a_batches_file)["a_batches"]
        else:
            print(f"Generating explanation maps for {method} and {pre_trained_model}")

            a_batches = []  # Store explanation maps for all batches
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(x_batch))

                x_batch_chunk = x_batch[start_idx:end_idx]
                y_batch_chunk = y_batch[start_idx:end_idx]

                # Generate explanations for the current batch
                if method == "Random":
                    a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method='Saliency', device=self.device)
                elif method == "GuidedGradCam":
                    layer = get_last_conv_layer(model)
                    a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method='GuidedGradCam', device=self.device, gc_layer=layer)
                else:
                    a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method=method, device=self.device)

                a_batches.append(a_batch)

            # Flatten the explanation maps
            a_batches = np.concatenate(a_batches, axis=0) if isinstance(a_batches[0], np.ndarray) else torch.cat(a_batches, dim=0).cpu().numpy()

            # Save the explanation maps to a compressed file
            os.makedirs(os.path.dirname(a_batches_file), exist_ok=True)
            np.savez_compressed(a_batches_file, a_batches=a_batches)
            print(f"Explanation maps saved to {a_batches_file}")

        a_masked_x_batch = []
        removed_list = []

        # choose the imc function based on the method and removal method (precentage wise or not)
        if precentage_wise:
            if method == "Random":
                imc_function = imc.percentage_random_remove
            else:
                imc_function = imc.percentage_remove
        else:
            if method == "Random":
                imc_function = imc.random_remove_pixels
            else:
                imc_function = imc.new_remove_pixels

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(x_batch))

            x_batch_chunk = x_batch[start_idx:end_idx]
            a_batch_chunk = a_batches[start_idx:end_idx]

            # Apply the masking function
            a_masked_chunk, removed = imc_function(a_batch_chunk, x_batch_chunk, threshold)

            a_masked_x_batch.extend(a_masked_chunk)
            removed_list.append(removed)
            
        # to flatten the list
        removed_list = [item for sublist in removed_list for item in sublist]



        # # Retrieve results and accuracy
        # df = imc.new_get_results5(a_masked_x_batch, y_batch, model, self.categories, removed_list)
        # Correctly = imc.get_corrects(df, self.top_k)
        # self.Correctly.append(Correctly)

        # # Save results to CSV
        # csv_dir = os.path.join(self.save_path, f"{method}", "csv")
        # os.makedirs(csv_dir, exist_ok=True)
        # df.to_csv(os.path.join(csv_dir, f"{method}_{threshold}_{pre_trained_model}.csv"), index=False)

        # Save images to the subfolder
        os.makedirs(subfolder_path, exist_ok=True)
        for imeg, label, i in zip(a_masked_x_batch, y_batch, range(len(a_masked_x_batch))):
            img = imc.change_ImageNet_format(imeg)
            img = Image.fromarray(img.astype('uint8'))
            imeg_label_idx = int(label.cpu().item()) if isinstance(label, torch.Tensor) else label
            real_name = self.categories[imeg_label_idx]
            img.save(os.path.join(subfolder_path, f"{i}_{real_name}.png"))

        removed = sum(removed_list) / len(removed_list)

        return f"{method} {threshold} {pre_trained_model} removed avrage: { removed } Correctly: (Correctly)"
    
    #  def filter_with_model_batch(self, threshold: float, method: str, pre_trained_model: str , precentage_wise: bool = False):
    #     """
    #     Filters the dataset using a pre-trained model and an explanation method in smaller batches.

    #     Args:
    #         threshold (float): The threshold value for removing pixels.
    #         method (str): The explanation method to use.
    #         pre_trained_model (str): The pre-trained model to use.

    #     Returns:
    #         str: A string containing the method, threshold, pre-trained model name, 
    #             the proportion of removed pixels, and the classification accuracy.
    #     """

       
    #     # Define the subfolder path for the current method, threshold, and pre-trained model
    #     subfolder_path = os.path.join(
    #         self.save_path, f"{method}", f"{method}_{threshold}_{pre_trained_model}"
    #     )

    #     # Check if the subfolder already exists
    #     if os.path.exists(subfolder_path):
    #         print(f"Skipping processing for {subfolder_path} as it already exists.")
    #         return f"Skipped {method} {threshold} {pre_trained_model}"

    #     # Proceed with processing if the subfolder does not exist
    #     x_link = os.path.join(self.data_path, "x_batch.pt")
    #     y_link = os.path.join(self.data_path, "y_batch.pt")

    #     # Load image batches
    #     x_batch, y_batch = imc.load_images(self.samples, x_link, y_link, self.device)

    #     # Load the pre-trained model
    #     exist_models = imc.exist_models()
    #     if pre_trained_model == exist_models[0]:
    #         model = imc.resnet18(self.device)
    #     elif pre_trained_model == exist_models[1]:
    #         model = imc.v3_small(self.device)
    #     elif pre_trained_model == exist_models[2]:
    #         model = imc.v3_large(self.device)
    #     elif pre_trained_model == exist_models[3]:
    #         model = imc.v3_inception(self.device)

    #     # Process in smaller batches
    #     batch_size = 32  # Adjust this value based on your system's memory capacity
    #     num_batches = (len(x_batch) + batch_size - 1) // batch_size
    #     x = len(x_batch)

    #     a_masked_x_batch = []
    #     removed_list = []

    #     # choose the imc function based on the method and removal method (precentage wise or not)
    #     if precentage_wise:
    #         if method == "Random":
    #             imc_function = imc.percentage_random_remove
    #         else:
    #             imc_function = imc.percentage_remove
    #     else:
    #         if method == "Random":
    #             imc_function = imc.random_remove_pixels
    #         else:
    #             imc_function = imc.new_remove_pixels

    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = min((i + 1) * batch_size, len(x_batch))

    #         x_batch_chunk = x_batch[start_idx:end_idx]
    #         y_batch_chunk = y_batch[start_idx:end_idx]

    #         # Generate explanations for the current batch
    #         if method == "Random":
    #             a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method='Saliency', device=self.device)
    #             a_masked_chunk, removed = imc_function(a_batch, x_batch_chunk, threshold)
    #         elif method == "GuidedGradCam":
    #             layer = get_last_conv_layer(model)
    #             a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method='GuidedGradCam', device=self.device , gc_layer=layer)
    #             a_masked_chunk, removed = imc_function(a_batch, x_batch_chunk, threshold)
    #         else:
    #             a_batch = quantus.explain(model, x_batch_chunk, y_batch_chunk, method=method, device=self.device)
    #             a_masked_chunk, removed  = imc_function(a_batch, x_batch_chunk, threshold)

    #         a_masked_x_batch.extend(a_masked_chunk)
    #         removed_list.append(removed)
            
    #     # to flatten the list
    #     removed_list = [item for sublist in removed_list for item in sublist]

    #     #categories = gd.load_imagenet_classes()


    #     # # Retrieve results and accuracy
    #     # df = imc.new_get_results5(a_masked_x_batch, y_batch, model, self.categories, removed_list)
    #     # Correctly = imc.get_corrects(df, self.top_k)
    #     # self.Correctly.append(Correctly)

    #     # # Save results to CSV
    #     # csv_dir = os.path.join(self.save_path, f"{method}", "csv")
    #     # os.makedirs(csv_dir, exist_ok=True)
    #     # df.to_csv(os.path.join(csv_dir, f"{method}_{threshold}_{pre_trained_model}.csv"), index=False)

    #     # Save images to the subfolder
    #     os.makedirs(subfolder_path, exist_ok=True)
    #     for imeg, label, i in zip(a_masked_x_batch, y_batch, range(len(a_masked_x_batch))):
    #         img = imc.change_ImageNet_format(imeg)
    #         img = Image.fromarray(img.astype('uint8'))
    #         imeg_label_idx = int(label.cpu().item()) if isinstance(label, torch.Tensor) else label
    #         real_name = self.categories[imeg_label_idx]
    #         img.save(os.path.join(subfolder_path, f"{i}_{real_name}.png"))

    #     removed = sum(removed_list) / len(removed_list)

    #     return f"{method} {threshold} {pre_trained_model} removed avrage: { removed } Correctly: (Correctly)"


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

        x_link = os.path.join(self.data_path, "x_batch.pt")
        y_link = os.path.join(self.data_path, "y_batch.pt")

        # Load image batches: Loads `nr_samples` samples for x (input images), y (labels), and s (saliency maps)
        # from the specified links and moves them to the given device (CPU or GPU).
        x_batch, y_batch = imc.load_images(self.samples, x_link, y_link, self.device)

        # Load the pre-trained model `v3_small` and assign it to the device for computation
        exist_models = imc.exist_models()
        if pre_trained_model == exist_models[0]:
            model = imc.resnet18(self.device)
        elif pre_trained_model == exist_models[1]:
            model = imc.v3_small(self.device)
        elif pre_trained_model == exist_models[2]:
            model = imc.v3_large(self.device)
        elif pre_trained_model == exist_models[3]:
            model = imc.v3_inception(self.device)


        # Generate explanations: Uses the `quantus.explain` function with the selected method
        # to calculate saliency maps (`a_batch`) based on the model’s predictions for `x_batch`.
        if method == "Random":
            a_batch = quantus.explain(model, x_batch, y_batch, method='Saliency', device=self.device)
            # Randomly remove pixels based on the threshold
            a_masked_x_batch, removed = imc.random_remove_pixels(a_batch, x_batch, threshold)
        else:
            a_batch = quantus.explain(model, x_batch, y_batch, method=method, device=self.device)
            # Remove pixels below the specified threshold in the explanation maps and calculate the masked x_batch.
            a_masked_x_batch, removed = imc.new_remove_pixels(a_batch, x_batch, threshold)

        # `a_masked_x_batch` is the result of applying the mask, and `removed` gives the proportion of pixels removed.

        self.removed.append(removed)

        # # Retrieve results and accuracy: Evaluate the top-k predictions based on the masked batch.
        # # The function `get_results` returns a DataFrame with results.
        # df = imc.new_get_results5(a_masked_x_batch, y_batch, model, self.categories)  # updated v2.0

        # # Calculate and print classification accuracy based on the top-k matches in `df`.
        # Correctly = imc.get_corrects(df, self.top_k)
        # self.Correctly.append(Correctly)

        # csv_dir = os.path.join( self.save_path, f"{method}", "csv")
        # os.makedirs(csv_dir, exist_ok=True)
        # df.to_csv(os.path.join(csv_dir, f"{method}_{threshold}_{pre_trained_model}.csv"), index=False)

        for imeg, label, i in zip(a_masked_x_batch, y_batch, range(len(a_masked_x_batch))):  # Use zip to iterate over both lists simultaneously
            # Format the image
            img = imc.change_ImageNet_format(imeg)
        
            # Convert the array to an image
            img = Image.fromarray(img.astype('uint8'))  # Ensure the data is in uint8 format for saving
    
            # Handle label if it's a Tensor
            imeg_label_idx = int(label.cpu().item()) if isinstance(label, torch.Tensor) else label

            real_name = self.categories[imeg_label_idx]


            # Save the image as PNG
            image_dir = os.path.join(self.save_path, f"{method}", f"{method}_{threshold}_{pre_trained_model}")
            os.makedirs(image_dir, exist_ok=True)
            img.save(os.path.join(image_dir, f"{i}_{real_name}.png"))  # Use forward slash or raw string literal for file paths

        return f"{method} {threshold} {pre_trained_model} removed: {removed} Correctly: -"
    
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

    
def get_last_conv_layer(model):
        """
        Dynamically find the last convolutional layer in the model.

        Args:
            model (torch.nn.Module): The model to search.

        Returns:
            torch.nn.Module: The last convolutional layer in the model.
        """
        for layer in reversed(list(model.modules())):
            if isinstance(layer, nn.Conv2d):
                return layer
        raise ValueError("No convolutional layer found in the model.")