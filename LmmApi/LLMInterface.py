from LmmApi.LLMStrategy import LLMStrategy
from data import global_data as gd
from LmmApi import global_lmm as glmm
from pydantic import BaseModel
import os
import pandas as pd
from tqdm import tqdm

# Context Class
class LLMInterface:


    def __init__(self, strategy: LLMStrategy):
        self.strategy = strategy
        self.imagenet_categories = gd.load_imagenet_classes()


    def set_strategy(self, strategy: LLMStrategy):
        """Set a different strategy at runtime."""
        self.strategy = strategy

    def set_prompt(self, prompt: str, jsonDescription: BaseModel | None = None):
        """Set the prompt for the LLM."""
        self.prompt = prompt
        self.jsonDescription = jsonDescription
    
    def get_response(self, image: str) -> str:
        """Generate a response using the current strategy."""
        return self.strategy.generate_response(self.prompt, image, self.jsonDescription)
         
    
    def process_response(self, response: list) -> str:
        """Process the response from the LLM."""
        arr = []
        for item in response:
            _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
            arr.append(closest_label)
        return arr


    def structured_outputs_classification(self, root_directory: str, save_path: str):
        """
        Classifies images in the specified root directory using a language model (LLM) strategy.
        The classification results are saved in CSV files, with one CSV file per method (folder name).

        Parameters:
        - root_directory (str): The root directory containing subdirectories with images to classify.
        - save_path (str): The directory where the CSV files with classification results will be saved.
        """
        os.makedirs(save_path, exist_ok=True)

        for dirpath, _, filenames in os.walk(root_directory): 
            image_files = [f for f in filenames if f.endswith('.png')] 
            data = []
            max_labels = 0
            max_llm = 0
            for file in tqdm(image_files, desc=f"Processing files in {dirpath}"):
                # Get the full path of the image file
                file_path = os.path.join(dirpath, file)

                # Generate a response from the LLM
                response = self.strategy.generate_response(self.prompt, file_path, self.jsonDescription)

                # Extract the model dump from the response
                response_list = list(response.model_dump().values())

                # Find the top 5 predicted labels
                labels = []
                for item in response_list:
                    _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
                    labels.append(closest_label)
                
                # Update the maximum number of labels
                max_labels = max(max_labels, len(labels))
                max_llm = max(max_llm, len(response_list))

                # Extract image index, true label, and method (directory name)
                index, mid = file.split('_')[:2]
                true_label = mid.split('.')[0]
                method = os.path.basename(dirpath)

                # Check if the true label is among the top 5 predicted labels
                correctly = true_label in labels

                data.append([index, true_label, correctly] + labels + response_list)

            if len(data) == 0:
                continue

            label_columns = [f'Class_{i+1}' for i in range(max_labels)]
            llm_columns = [f'llm_{i+1}' for i in range(max_llm)]
            columns = ['Index', 'True_Label', 'Match'] + label_columns + llm_columns
            # Create a DataFrame from the array and add the column titles
            df = pd.DataFrame(data, columns=columns)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(save_path, f"{method}.csv"), index=False)




                


