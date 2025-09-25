from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
#from LmmApi.chatGpt4o import ChatGPT4O
from trash import dynamic_results as dr
# from dataset_API import dataset_interface as di
# from dataset_API import image_creater as imc
import os
from pydantic import BaseModel

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

    llm_context.boolean_outputs_classification_reverse("data\\midAvi_grey", "data\\llm_answer_Avi_grey")
