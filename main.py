from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
from data  import global_data as gd
from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc
import os
from pydantic import BaseModel
from typing import Literal
from typing import Union
import pandas as pd



def calculate_accuracy(root_directory: str, save_path: str):

    #os.makedirs(save_path, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_directory): 
        image_files = [f for f in filenames if f.endswith('.csv')] 

        for file in image_files:
            # Get the full path of the image file
            file_path = os.path.join(dirpath, file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Calculate the average of the 'Match' column
            average_match = df['Match'].mean()

            print(f"Average match for {file}: {average_match}")




def ask_llm(imges_path: str, save_path: str, jsonDescription: BaseModel | None = None):
    llama = llama32Vision11b()
    llm_context = LLMInterface(llama)
    llm_context.set_prompt("Tell me what you see in the picture and what category it is from imagenet", jsonDescription)

    llm_context.anchored_outputs_classification(imges_path, save_path)
    return llm_context 


def image_creater(dir_path: str, save_path: str, samples: int = 200):
    """
    Generates a dataset by applying filters using pre-trained models and explanation methods.

    Args:
        dir_path (str): Path to the directory containing the input dataset.
        save_path (str): Path where the filtered dataset will be saved.
        samples (int, optional): The number of samples to process. Default is 200.

    Returns:
        object: The dataset object after applying filters.

    Behavior:
        - If the folder specified by `save_path` already exists, the function exits early and returns.
        - Otherwise, it processes the dataset with various thresholds using pre-trained models
          and explanation methods, saving the results to `save_path`.
    """
    # Check if the folder already exists
    if os.path.exists(save_path):
        print(f"The folder '{save_path}' already exists. Returning without processing.")
        return

    # Load pre-trained models
    pre_trained_model = imc.exist_models()

    # Explanation methods and thresholds
    explanation_methods = ['Saliency']
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # Initialize dataset interface
    dataset = di.dataset_interface(dir_path, save_path, samples)

    # Apply filters using models, explanation methods, and thresholds
    for prtm in pre_trained_model:
        for expm in explanation_methods:
            for thresh in thresholds:
                dataset.filter_with_model(thresh, expm, prtm)

    return dataset

#imagenet_classes = gd.load_imagenet_classes()

class ImageDescription_1(BaseModel):
    Summary: str
    category: str

class ImageDescription_5(BaseModel):
    category1: str
    category2: str
    category3: str
    category4: str
    category5: str
    


if __name__ == "__main__":
    image_creater("data\\source\\imagenet_sample2\\pt", "data\\mid", samples=500)

    #ask_llm("data\\mid\\Saliency_0.001_resnet18", "data\\llm_answer\\anchored_structured_outputs\\5_categoris", ImageDescription_5)
    #ask_llm("data\\mid\\Saliency_0.001_resnet18", "data\\llm_answer\\anchored_structured_outputs\\1_categoris", ImageDescription_1)

    calculate_accuracy("data\\llm_answer\\anchored_structured_outputs\\5_categoris", "data\\llm_answer\\structured_outputs\\5_categoris")
    print("-------------------------------------------------")
    calculate_accuracy("data\\llm_answer\\anchored_structured_outputs\\1_categoris", "data\\llm_answer\\structured_outputs\\1_categoris")
