from data  import dynamic_results as dr
from data  import results

if __name__ == "__main__":

    # dr.getDynamicResults("InputXGradient", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.getDynamicResults("GradientShap", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.getDynamicResults("Random", "boolean", "llm_answer_precentage", "midPrecentage")

    # results.calculate_accuracy("data\\llm_answer_precentage\\InputXGradient\\boolean","data\\llm_answer_precentage\\InputXGradient\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_precentage\\GradientShap\\boolean","data\\llm_answer_precentage\\GradientShap\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_precentage\\GuidedGradCam\\boolean","data\\llm_answer_precentage\\GuidedGradCam\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\Random\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\Random\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\Saliency\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\Saliency\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\GuidedGradCam\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\GuidedGradCam\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\InputXGradient\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\InputXGradient\\boolean\\results")

    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE_reversed\\Saliency\\boolean", "data\\llm_answer_CervicalCancer_COMPLETE_reversed\\Saliency\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE_reversed\\Random\\boolean", "data\\llm_answer_CervicalCancer_COMPLETE_reversed\\Random\\boolean\\results")
    # results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE_reversed\\InputXGradient\\boolean", "data\\llm_answer_CervicalCancer_COMPLETE_reversed\\InputXGradient\\boolean\\results")

    results.calculate_accuracy("data\\midPrecentage\\GradientShap", "data\\midPrecentage\\GradientShap\\results")
    results.calculate_accuracy("data\\midPrecentage\\GuidedGradCam", "data\\midPrecentage\\GuidedGradCam\\results")
    results.calculate_accuracy("data\\midPrecentage\\InputXGradient", "data\\midPrecentage\\InputXGradient\\results")
    results.calculate_accuracy("data\\midPrecentage\\Random", "data\\midPrecentage\\Random\\results")
    results.calculate_accuracy("data\\midPrecentage\\Saliency", "data\\midPrecentage\\Saliency\\results")
