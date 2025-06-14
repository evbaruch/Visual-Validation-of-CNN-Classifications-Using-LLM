
from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc
from data import ImageNetProcessor as inp
from data  import results

def image_creater(dir_path: str, labels_decoded: list, save_path: str, samples: int = 10 ,precentage_wise: bool = False):
    # Load pre-trained models
    pre_trained_model = imc.exist_models()

    # Explanation methods and thresholds
    #  ['GradientShap', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap', 'InputXGradient', 'Saliency', 'FeatureAblation', 'Deconvolution', 'FeaturePermutation', 'Lime', 'KernelShap', 'LRP', 'Gradient', 'Occlusion', 'LayerGradCam', 'GuidedGradCam', 'LayerConductance', 'LayerActivation', 'InternalInfluence', 'LayerGradientXActivation', 'Control Var. Sobel Filter', 'Control Var. Constant', 'Control Var. Random Uniform']
    explanation_methods = ['Random' , 'Saliency', 'GuidedGradCam', 'InputXGradient', 'GradientShap'] # 'Lime', 'GuidedGradCam', 'InputXGradient',

    thresholds = [0.001, 0.002, 0.005, 0.01, 0.02,  0.05,  0.1, 0.2, 0.5]
    precentages = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95]
    #precentages = [5, 20, 50, 70, 95]
    backgroundColors = [0, 0.5, 1]
    reverses = [False, True]

    # Initialize dataset interface
    dataset = di.dataset_interface(dir_path, save_path, labels_decoded, samples)

    # Apply filters using models, explanation methods, and thresholds
    for expm in explanation_methods:
        for prtm in pre_trained_model:
            for precentage in precentages:
                for color in backgroundColors:
                    for reverse in reverses:
                        dataset.filter_with_model_batch(precentage, expm, prtm, precentage_wise, color, reverse)

    return dataset



if __name__ == "__main__":

    imagenet = inp.ImageNetProcessor()
    imagenet.save_tensors()
    
    image_creater(imagenet.pt_dir, imagenet.labels_decoded, "data\\midImagenetTEST", 200 , True)
    
    results.calculate_accuracy("data\\midImagenetTEST\\GuidedGradCam\\csv","data\\midImagenetTEST\\GuidedGradCam\\csv\\results")
    results.calculate_accuracy("data\\midImagenetTEST\\GradientShap\\csv","data\\midImagenetTEST\\GradientShap\\csv\\results")
    results.calculate_accuracy("data\\midImagenetTEST\\InputXGradient\\csv","data\\midImagenetTEST\\InputXGradient\\csv\\results")
    results.calculate_accuracy("data\\midImagenetTEST\\Random\\csv","data\\midImagenetTEST\\Random\\csv\\results")
    results.calculate_accuracy("data\\midImagenetTEST\\Saliency\\csv","data\\midImagenetTEST\\Saliency\\csv\\results")

