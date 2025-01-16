from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
from data  import results
from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc
import os
from pydantic import BaseModel
import pandas as pd





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
    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02,  0.05,  0.1, 0.2, 0.5]

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

class ImageDescription_Boolean(BaseModel):
    boolean: bool   
    


if __name__ == "__main__":
    image_creater("data\\source\\imagenet_sample2\\pt", "data\\mid", samples=200)

    llama = llama32Vision11b()
    llm_context = LLMInterface(llama)
    llm_context.set_prompt("Tell me what you see in the picture and what category it is from imagenet")


    llm_context.set_jsonDescription(ImageDescription_Boolean)
    llm_context.boolean_outputs_classification("data\\mid", "data\\llm_answer\\boolean")

    results.calculate_accuracy("data\\llm_answer\\boolean", "data\\llm_answer\\boolean\\results")
    #results.calculate_accuracy("data\\mid\\csv", "data\\mid\\csv\\results")
