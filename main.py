from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
#from LmmApi.chatGpt4o import ChatGPT4O
from data  import results
from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc
import os
from pydantic import BaseModel


# def ask_llm(imges_path: str, save_path: str, jsonDescription: BaseModel):
#     llama = llama32Vision11b()
#     llm_context = LLMInterface(llama)
#     llm_context.set_prompt("Tell me what you see in the picture and what category it is from imagenet", jsonDescription)

#     llm_context.anchored_outputs_classification(imges_path, save_path)
#     return llm_context 


def image_creater(dir_path: str, save_path: str, samples: int = 10):
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
    # if os.path.exists(save_path):
    #     print(f"The folder '{save_path}' already exists. Returning without processing.")
    #     return

    # Load pre-trained models
    pre_trained_model = imc.exist_models()

    # Explanation methods and thresholds
    #  ['GradientShap', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap', 'InputXGradient', 'Saliency', 'FeatureAblation', 'Deconvolution', 'FeaturePermutation', 'Lime', 'KernelShap', 'LRP', 'Gradient', 'Occlusion', 'LayerGradCam', 'GuidedGradCam', 'LayerConductance', 'LayerActivation', 'InternalInfluence', 'LayerGradientXActivation', 'Control Var. Sobel Filter', 'Control Var. Constant', 'Control Var. Random Uniform']
    explanation_methods = ['Random', 'GradientShap',  'Saliency']  # 'Lime', 'GuidedGradCam', 'InputXGradient',
    do_explanation_methods = []

    # Temporary! It doesn't make sense for this test to be implemented outside of dataset_interface === TO DO ===
    for exp in explanation_methods:
        if not os.path.exists(os.path.join(save_path, exp)):
            do_explanation_methods.append(exp)


    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02,  0.05,  0.1, 0.2, 0.5]

    # Initialize dataset interface
    dataset = di.dataset_interface(dir_path, save_path, samples)

    # Apply filters using models, explanation methods, and thresholds
    for expm in do_explanation_methods:
        for prtm in pre_trained_model:
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
    # image_creater("data\\source\\imagenet_sample2\\pt", "data\\mid", samples=10)

    llama = llama32Vision11b()
    # llama = ChatGPT4O("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")
    llm_context = LLMInterface(llama)
    # llm_context.set_prompt("Tell me what you see in the picture and what category it is from imagenet")

    llm_context.set_jsonDescription(ImageDescription_Boolean)
    llm_context.boolean_outputs_classification("data\\mid\\Random", "data\\llm_answer2\\Random\\boolean")

    llm_context.set_jsonDescription(ImageDescription_Boolean)
    llm_context.boolean_outputs_classification("data\\mid\\Saliency", "data\\llm_answer2\\Saliency\\boolean")

    llm_context.set_jsonDescription(ImageDescription_Boolean)
    llm_context.boolean_outputs_classification("data\\mid\\GradientShap", "data\\llm_answer2\\GradientShap\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_5)
    # llm_context.anchored_outputs_classification("data\\mid\\Random", "data\\llm_answer\\Random\\anchored_structured_outputs\\5_categoris")

    # llm_context.set_jsonDescription(ImageDescription_1)
    # llm_context.anchored_outputs_classification("data\\mid\\Random", "data\\llm_answer\\Random\\anchored_structured_outputs\\1_categoris")

    # llm_context.structured_outputs_classification("data\\mid", "data\\llm_answer\\structured_outputs\\1_categoris")



    # results.calculate_accuracy("data\\llm_answer\\anchored_structured_outputs\\5_categoris","data\\llm_answer\\anchored_structured_outputs\\5_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\anchored_structured_outputs\\1_categoris","data\\llm_answer\\anchored_structured_outputs\\1_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\structured_outputs\\5_categoris","data\\llm_answer\\structured_outputs\\5_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\structured_outputs\\1_categoris","data\\llm_answer\\structured_outputs\\1_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\boolean","data\\llm_answer\\boolean\\results")
    # results.calculate_accuracy("data\\mid\\Random\\csv","data\\mid\\Random\\csv\\results")
    # results.calculate_accuracy("data\\mid\\Saliency\\csv","data\\mid\\Saliency\\csv\\results")
    # results.calculate_accuracy("data\\mid\\GradientShap\\csv","data\\mid\\GradientShap\\csv\\results")
    # results.calculate_accuracy("data\\llm_answer\\Random\\anchored_structured_outputs\\5_categoris","data\\llm_answer\\Random\\anchored_structured_outputs\\5_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\Random\\anchored_structured_outputs\\1_categoris","data\\llm_answer\\Random\\anchored_structured_outputs\\1_categoris\\results")
    # results.calculate_accuracy("data\\llm_answer\\Random\\boolean","data\\llm_answer\\Random\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer\\InputXGradient\\boolean","data\\llm_answer\\InputXGradient\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer\\GradientShap\\boolean","data\\llm_answer\\GradientShap\\boolean\\results")