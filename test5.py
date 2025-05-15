if __name__ == "__main__":

    #image_creater("data/source/CervicalCancer/pt/CROPPED_40", "data\\midCervicalCancer", 200000, True) 
    image_creater("data\\source\\CervicalCancer\\pt\\COMPLETE_20", "data\midCervicalCancer_COMPLETE_reversed", 2000 , True)

    llama = llama32Vision11b()
    # # # llama = ChatGPT4O("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")
    llm_context = LLMInterface(llama)

    
    llm_context.set_background(r"""You are a medical image analysis expert specialized in cytopathology. You are tasked with classifying microscopic images of cervical cells into one of the following categories based on their visual characteristics: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, or Superficial-Intermediate.

    Each cell type has unique morphological features:
    - Dyskeratotic: Abnormal keratinization, hyperchromatic nuclei.
    - Koilocytotic: Perinuclear halo, nuclear enlargement, irregularity.
    - Metaplastic: Immature squamous cells, dense cytoplasm.
    - Parabasal: Small round cells with large nuclei, usually in clusters.
    - Superficial-Intermediate: Flattened cells with small nuclei, abundant cytoplasm.

    Use your visual recognition capabilities and domain expertise to classify each provided image.
    return as JSON!
    """)

    # # llm_context.set_background(f"You are an image classifier. Use the ImageNet categories to classify images. return as JSON")

    # # # llm_context.set_prompt("Tell me what you see in the picture and  what category it is from imagenet")

    llm_context.set_jsonDescription(ImageDescription_Boolean)
    llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE_reversed\\Saliency", "data\\midCervicalCancer_COMPLETE_reversed\\Saliency\\boolean")

    llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE_reversed\\Random", "data\\midCervicalCancer_COMPLETE_reversed\\Random\\boolean")

    llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE_reversed\\InputXGradient", "data\\midCervicalCancer_COMPLETE_reversed\\InputXGradient\\boolean")

    llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE_reversed\\GuidedGradCam", "data\\midCervicalCancer_COMPLETE_reversed\\GuidedGradCam\\boolean")

    llm_context.boolean_outputs_classification("data\\midCervicalCancer_COMPLETE_reversed\\GradientShap", "data\\midCervicalCancer_COMPLETE_reversed\\GradientShap\\boolean")


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

explainability_mode = ['reversed' , 'normal']

#---------------------------## main file ##----------------------------#
classification_details = {
    0: [path_builder('midCervicalCancer_COMPLETE_reversed' , "Saliency" , 'boolean') , 'boolean'), 
    path_builder('midCervicalCancer_COMPLETE_reversed' , "Saliency" , 'boolean' , 'results'),
    explainability_mode[0]]
}


foo(classification_details[0][0] , classification_details[0][1] , classification_details[0][2])


#---------------------------### setup file ###---------------------------#
def foo(save_path , load_path, explainability_mode ):

    llm_context.boolean_outputs_classification(load_path , save_path)
    # all actions of our research are in this function
    results.calculate_accuracy("data\\llm_answer2\\Saliency\\boolean","data\\llm_answer2\\Saliency\\boolean\\results")

def path_builder(base_dir , explainability_method , query_mode = None ,sub_dir = None):
    """
    this function will build the path for the foo method

    1. All data is in the dir \data
    2. The base_dir is the base directory in the data folder (mid , llm_answer , midCervicalCancer , llm_answer2 , ect.)
    3. The explainability_method is the method used to explain the image (Saliency , Random , GradientShap , InputXGradient , GuidedGradCam)
    4. The query_mode is the mode used to query the image (boolean , anchored_structured_outputs , structured_outputs)
    """
    save_path = os.path.join("data", base_dir, explainability_method, query_mode, sub_dir)
    return save_path

##---------------------------## image prossening file ##----------------------------#

def 