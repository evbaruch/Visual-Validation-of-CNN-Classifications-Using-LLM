from dataset_API import dataset_interface as di

a = di.dataset_interface("data\\source\\imagenet-sample2\\pt", "data\\mid")

a.filter_with_model(0.01, 'Saliency', "v3_small")