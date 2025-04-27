from data import dynamic_results as dr
from data import results

dr.getDynamicResults("Saliency", "boolean", "llm_answer_precentage", "midPrecentage")
dr.getDynamicResults("Random", "boolean", "llm_answer_precentage", "midPrecentage")

results.calculate_accuracy("data\\llm_answer_precentage\\Random\\boolean","data\\llm_answer_precentage\\Random\\boolean\\results")
results.calculate_accuracy("data\\llm_answer_precentage\\Saliency\\boolean","data\\llm_answer_precentage\\Saliency\\boolean\\results")