from data  import dynamic_results as dr
from data  import results

if __name__ == "__main__":

    # dr.getDynamicResults("InputXGradient", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.getDynamicResults("GradientShap", "boolean", "llm_answer_precentage", "midPrecentage")
    # dr.getDynamicResults("Random", "boolean", "llm_answer_precentage", "midPrecentage")

    results.calculate_accuracy("data\\llm_answer_precentage\\InputXGradient\\boolean","data\\llm_answer_precentage\\InputXGradient\\boolean\\results")
    results.calculate_accuracy("data\\llm_answer_precentage\\GradientShap\\boolean","data\\llm_answer_precentage\\GradientShap\\boolean\\results")
    results.calculate_accuracy("data\\llm_answer_precentage\\GuidedGradCam\\boolean","data\\llm_answer_precentage\\GuidedGradCam\\boolean\\results")
    results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\Random\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\Random\\boolean\\results")
    results.calculate_accuracy("data\\llm_answer_CervicalCancer_COMPLETE\\Saliency\\boolean","data\\llm_answer_CervicalCancer_COMPLETE\\Saliency\\boolean\\results")
0000000