from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
#from LmmApi.chatGpt4o import ChatGPT4O
from data  import results
from data import dynamic_results as dr
from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc
import os
from pydantic import BaseModel
# import kagglehub
# import shutil
# from data import CCDataSet_init as CCD

# def ask_llm(imges_path: str, save_path: str, jsonDescription: BaseModel):
#     llama = llama32Vision11b()
#     llm_context = LLMInterface(llama)
#     llm_context.set_prompt("Tell me what you see in the picture and what category it is from imagenet", jsonDescription)

#     llm_context.anchored_outputs_classification(imges_path, save_path)
#     return llm_context 


def image_creater(dir_path: str, save_path: str, samples: int = 10 ,precentage_wise: bool = False):
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
    explanation_methods = ['Random' , 'Saliency', 'GuidedGradCam', 'InputXGradient', 'GradientShap'] # 'Lime', 'GuidedGradCam', 'InputXGradient',

    # # Temporary! It doesn't make sense for this test to be implemented outside of dataset_interface === TO DO ===
    # for exp in explanation_methods:
    #     if not os.path.exists(os.path.join(save_path, exp)):
    #         do_explanation_methods.append(exp)


    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02,  0.05,  0.1, 0.2, 0.5]
    precentages = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

    # Initialize dataset interface
    dataset = di.dataset_interface(dir_path, save_path, samples)

    # Apply filters using models, explanation methods, and thresholds
    if precentage_wise:
        for expm in explanation_methods:
            for prtm in pre_trained_model:
                for precentage in precentages:
                    dataset.filter_with_model_batch(precentage, expm, prtm, precentage_wise)
    else:
        for expm in explanation_methods:
            for prtm in pre_trained_model:
                for thresh in thresholds:
                    dataset.filter_with_model_batch(thresh, expm, prtm)

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

    #image_creater("data/source/CervicalCancer/pt/CROPPED_40", "data\\midCervicalCancer", 200000, True) 
    image_creater("data\\source\\CervicalCancer\\pt\\COMPLETE_20", "data\midCervicalCancer_COMPLETE_reversed", 2000 , True)

    # llama = llama32Vision11b()
    # # # llama = ChatGPT4O("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")
    # llm_context = LLMInterface(llama)

    
    # llm_context.set_background(r"""You are a medical image analysis expert specialized in cytopathology. You are tasked with classifying microscopic images of cervical cells into one of the following categories based on their visual characteristics: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, or Superficial-Intermediate.

    # Each cell type has unique morphological features:
    # - Dyskeratotic: Abnormal keratinization, hyperchromatic nuclei.
    # - Koilocytotic: Perinuclear halo, nuclear enlargement, irregularity.
    # - Metaplastic: Immature squamous cells, dense cytoplasm.
    # - Parabasal: Small round cells with large nuclei, usually in clusters.
    # - Superficial-Intermediate: Flattened cells with small nuclei, abundant cytoplasm.

    # Use your visual recognition capabilities and domain expertise to classify each provided image.
    # return as JSON!
    # """)

    # # llm_context.set_background(f"You are an image classifier. Use the ImageNet categories to classify images. return as JSON")

    # # # llm_context.set_prompt("Tell me what you see in the picture and  what category it is from imagenet")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE\\GuidedGradCam", "data\\llm_answer_CervicalCancer_COMPLETE\\GuidedGradCam\\boolean")

    # llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE\\InputXGradient", "data\\llm_answer_CervicalCancer_COMPLETE\\InputXGradient\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midCervicalCancer\\Saliency", "data\\llm_answer_CervicalCancer\\Saliency\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midsample2\\Random", "data\\llm_answer2\\Random\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midsample2\\GradientShap", "data\\llm_answer2\\GradientShap\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midsample2\\InputXGradient", "data\\llm_answer2\\InputXGradient\\boolean")

    # llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midsample2\\GuidedGradCam", "data\\llm_answer2\\GuidedGradCam\\boolean")

    # dr.add_precentage_to_csv("GradientShap", "boolean", "llm_answer2", "midsample2")
    # dr.getDynamicResults("Random", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.getDynamicResults("Saliency", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.add_precentage_to_csv("Random", "boolean", "llm_answer_precentage", "midPrecentage")

    # dr.add_precentage_to_csv("Saliency", "boolean", "llm_answer2", "midsample2")
    # dr.add_precentage_to_csv("InputXGradient", "boolean", "llm_answer2", "midsample2")
    # dr.add_precentage_to_csv("GuidedGradCam", "boolean", "llm_answer2", "midsample2")
    
    # dr.calculate_accuracy("data\\llm_answer_precentage\\Random\\boolean", "data\\llm_answer_precentage\\Random\\boolean\\p_results")
    
    # dr.calculate_accuracy("data\\llm_answer2\\Saliency\\boolean", "data\\llm_answer2\\Saliency\\boolean\\p_results")
    # dr.calculate_accuracy("data\\llm_answer2\\GradientShap\\boolean", "data\\llm_answer2\\GradientShap\\boolean\\p_results")
    # dr.calculate_accuracy("data\\llm_answer2\\InputXGradient\\boolean", "data\\llm_answer2\\InputXGradient\\boolean\\p_results")
    # dr.calculate_accuracy("data\\llm_answer2\\GuidedGradCam\\boolean", "data\\llm_answer2\\GuidedGradCam\\boolean\\p_results")

    # results.calculate_accuracy("data\\llm_answer2\\Saliency\\boolean","data\\llm_answer2\\Saliency\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_precentage\\Random\\boolean","data\\llm_answer_precentage\\Random\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_precentage\\Saliency\\boolean","data\\llm_answer_precentage\\Saliency\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer2\\GradientShap\\boolean","data\\llm_answer2\\GradientShap\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer2\\InputXGradient\\boolean","data\\llm_answer2\\InputXGradient\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer2\\GuidedGradCam\\boolean","data\\llm_answer2\\GuidedGradCam\\boolean\\results")

    # dr.transform_results_to_table("data\\llm_answer_precentage\\Random\\boolean\\p_results\\results.csv", "data\\llm_answer_precentage\\Random\\boolean\\p_results","Random")
    
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