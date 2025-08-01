from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
#from LmmApi.chatGpt4o import ChatGPT4O
from trash import dynamic_results as dr
# from dataset_API import dataset_interface as di
# from dataset_API import image_creater as imc
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


# def image_creater(dir_path: str, save_path: str, samples: int = 10 ,precentage_wise: bool = False):
#     """
#     Generates a dataset by applying filters using pre-trained models and explanation methods.

#     Args:
#         dir_path (str): Path to the directory containing the input dataset.
#         save_path (str): Path where the filtered dataset will be saved.
#         samples (int, optional): The number of samples to process. Default is 200.

#     Returns:
#         object: The dataset object after applying filters.

#     Behavior:
#         - If the folder specified by `save_path` already exists, the function exits early and returns.
#         - Otherwise, it processes the dataset with various thresholds using pre-trained models
#           and explanation methods, saving the results to `save_path`.
#     """
#     # Check if the folder already exists
#     # if os.path.exists(save_path):
#     #     print(f"The folder '{save_path}' already exists. Returning without processing.")
#     #     return

#     # Load pre-trained models
#     pre_trained_model = imc.exist_models()

#     # Explanation methods and thresholds
#     #  ['GradientShap', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap', 'InputXGradient', 'Saliency', 'FeatureAblation', 'Deconvolution', 'FeaturePermutation', 'Lime', 'KernelShap', 'LRP', 'Gradient', 'Occlusion', 'LayerGradCam', 'GuidedGradCam', 'LayerConductance', 'LayerActivation', 'InternalInfluence', 'LayerGradientXActivation', 'Control Var. Sobel Filter', 'Control Var. Constant', 'Control Var. Random Uniform']
#     explanation_methods = ['Random' , 'Saliency', 'GuidedGradCam', 'InputXGradient', 'GradientShap'] # 'Lime', 'GuidedGradCam', 'InputXGradient',

#     # # Temporary! It doesn't make sense for this test to be implemented outside of dataset_interface === TO DO ===
#     # for exp in explanation_methods:
#     #     if not os.path.exists(os.path.join(save_path, exp)):
#     #         do_explanation_methods.append(exp)


#     thresholds = [0.001, 0.002, 0.005, 0.01, 0.02,  0.05,  0.1, 0.2, 0.5]
#     precentages = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]

#     # Initialize dataset interface
#     dataset = di.dataset_interface(dir_path, save_path, samples)

#     # Apply filters using models, explanation methods, and thresholds
#     if precentage_wise:
#         for expm in explanation_methods:
#             for prtm in pre_trained_model:
#                 for precentage in precentages:
#                     dataset.filter_with_model_batch(precentage, expm, prtm, precentage_wise)
#     else:
#         for expm in explanation_methods:
#             for prtm in pre_trained_model:
#                 for thresh in thresholds:
#                     dataset.filter_with_model_batch(thresh, expm, prtm)

#     return dataset

#imagenet_classes = gd.load_imagenet_classes()


class ImageDescription_Boolean(BaseModel):
    boolean: bool   
    
 


if __name__ == "__main__":

    llama = llama32Vision11b()
    llm_context = LLMInterface(llama)

    
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

    llm_context.set_background(f"You are an image classifier. Use the ImageNet categories to classify images. return as JSON")


    llm_context.set_jsonDescription(ImageDescription_Boolean)
    # llm_context.boolean_outputs_classification("data\\midAvi_black\\gradcam", "data\\llm_answer_Avi_black\\gradcam\\boolean")

    # llm_context.boolean_outputs_classification("data\\midAvi_black\\random", "data\\llm_answer_Avi_black\\Random\\boolean")
    
    # llm_context.boolean_outputs_classification("data\\midAvi_grey\\random", "data\\llm_answer_Avi_grey\\random\\boolean")

    # llm_context.boolean_outputs_classification("data\\midAvi_grey\\gradcam", "data\\llm_answer_Avi_grey\\gradcam\\boolean")
    
    # ["guided_backprop","guided_gradcam", "integrated_gradients","inputxgradient", "smoothgrad"]
    llm_context.boolean_outputs_classification_reverse2("data\\midAvi_grey", "data\\llm_answer_Avi_grey")
    # llm_context.boolean_outputs_classification("data\\midAvi_grey\\gradientshap", "data\\llm_answer_Avi_grey\\gradientshap\\boolean")
    
    # llm_context.boolean_outputs_classification("data\\midAvi_grey\\random", "data\\llm_answer_Avi_grey\\random2\\boolean")


    # results.calculate_accuracy("data\\llm_answer_Avi\\gradcam\\boolean", "data\\results_Avi\\gradcam")
    # results.calculate_accuracy("data\\llm_answer_Avi\\Random\\boolean", "data\\results_Avi\\random")

    # results.calculate_accuracy("data\\llm_answer_Avi_grey\\gradcam\\boolean", "data\\results_Avi_grey\\gradcam")
    # results.calculate_accuracy("data\\llm_answer_Avi_grey\\Random\\boolean", "data\\results_Avi_grey\\random")
    
    # results.calculate_accuracy("data\\llm_answer_Avi_grey\\saliency\\boolean", "data\\results_Avi_grey\\saliency")
    # results.calculate_accuracy("data\\llm_answer_Avi_grey\\gradientshap\\boolean", "data\\results_Avi_grey\\gradientshap")
    
    #results.calculate_accuracy("data\\llm_answer_Avi_grey\\random2\\boolean", "data\\results_Avi_grey\\random2")
