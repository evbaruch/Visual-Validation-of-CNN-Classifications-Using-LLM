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
        self.response = self.strategy.generate_response(self.prompt, image, self.jsonDescription)
        return self.response
    
    def process_response(self, response: list) -> str:
        """Process the response from the LLM."""
        arr = []
        for item in response:
            _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
            arr.append(closest_label)

        return arr


    def classify_images_with_llm(self, root_directory: str, save_path: str):
        for dirpath, _, filenames in os.walk(root_directory):
            data = []
            for file in tqdm(filenames, desc=f"Processing files in {dirpath}"):
                if file.endswith('.png'):
                    file_path = os.path.join(dirpath, file)

                response = self.strategy.generate_response(self.prompt, file_path, self.jsonDescription)

                response_list = list(response.model_dump().values())

                labels = []
                for item in response_list:
                    _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
                    labels.append(closest_label)

                index = file.split('_')[0]
                mid = file.split('_')[1]
                True_label = mid.split('.')[0]
                method = dirpath.split('\\')[-1]

                correctly = False
                if True_label in labels:
                    correctly = True

                data.append([index, True_label] + labels + [correctly])

            columns = ['Index', 'True_Label', 'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5', 'Correctly']
            # Create a DataFrame from the array and add the column titles
            df = pd.DataFrame(data, columns=columns)



            os.makedirs(save_path, exist_ok=True)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(save_path, f"{method}.csv"), index=False)




                


