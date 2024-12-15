from dataset_API import dataset_interface as di
from dataset_API import image_creater as imc

pre_trained_model = imc.exist_models()

explanation_methods = ['Saliency']

thresholds = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

dataset = di.dataset_interface("data\\source\\imagenet-sample2\\pt", "data\\mid")

arr = []
for prtm in pre_trained_model:
    for expm in explanation_methods:
        for thresh in thresholds:
            a = dataset.filter_with_model(thresh, expm, prtm)
            arr.append(a)

for a in arr:
    print(a)
