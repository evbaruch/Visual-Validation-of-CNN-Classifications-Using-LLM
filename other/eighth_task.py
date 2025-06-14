import requests
import torch
import torchvision
from typing import Union
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
from IPython.display import clear_output
from IPython.display import Audio
from IPython.core.display import display
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import quantus

clear_output()

"""## Models"""

# Returns a list of model names that can be used in the code.
def exist_models():
    return ["resnet18()", "v3_small()", "v3_large()", "v3_inception()"]

# Loads a pre-trained ResNet-18 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def resnet18():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained MobileNetV3-Small model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_small():
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained MobileNetV3-Large model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_large():
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

# Loads a pre-trained InceptionV3 model, moves it to the GPU if available, sets it to evaluation mode, and returns the model and device.
def v3_inception():
    model = torchvision.models.inception_v3(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    clear_output()
    return model, device

"""## Images"""

# Load images from a .npy file, extracting x_batch, y_batch, and s_batch for the specified number of images.
def load_images(num, link):
    assets = np.load(link, allow_pickle=True).item()
    x_batch = assets["x_batch"][:num]
    y_batch = assets["y_batch"][:num]
    s_batch = assets["s_batch"].reshape(-1, 1, 224, 224)[:num]
    return x_batch, y_batch, s_batch

# Load images from separate .pt files for x_batch, y_batch, and s_batch, and move them to the specified device.
def load_images(num, x_link, y_link, s_link, device):
    x_batch = torch.load(x_link).to(device)[:num]
    y_batch = torch.load(y_link).to(device)[:num]
    s_batch = torch.load(s_link).reshape(-1, 1, 224, 224).to(device)[:num]
    return x_batch, y_batch, s_batch

"""## Utils"""

# Denormalize an image from the ImageNet format and convert it to a format suitable for display.
def normalize_image(arr: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if isinstance(arr, torch.Tensor):
        arr_copy = arr.clone().cpu().numpy()
    else:
        arr_copy = arr.copy()

    # arr_copy = (np.array(arr_copy) * std.reshape(-1, 1, 1)) + mean.reshape(-1, 1, 1)
    arr_copy  = np.moveaxis(arr_copy, 0, -1)
    arr_copy = (arr_copy * 255.).astype(np.uint8)
    return arr_copy

# Mask pixels in x_batch based on the threshold applied to a_batch, setting pixels below the threshold to 1.
def remove_pixels(a_batch, x_batch, threshold):
  a_copied_matrix = np.copy(a_batch)
  a_copied_matrix[a_copied_matrix <= threshold] = 0  # Set pixels below threshold to 0
  a_mask = (a_copied_matrix != 0)
  a_reshaped_mask = np.repeat(a_mask, x_batch.shape[1], axis=1) ## מיותר???
  a_masked_x_batch = np.copy(x_batch)
  a_masked_x_batch[~a_reshaped_mask] = 1

  total_pixels = np.prod(x_batch.shape)
  removed_pixels = np.sum(~a_reshaped_mask)

  return a_masked_x_batch, removed_pixels / total_pixels

# Improved version of `remove_pixels`, compatible with both torch tensors and numpy arrays.
def new_remove_pixels(a_batch, x_batch, threshold):
    # Ensure a_batch and x_batch are both NumPy arrays for compatibility with NumPy operations
    if isinstance(a_batch, torch.Tensor):
        a_batch = a_batch.cpu().numpy()
    if isinstance(x_batch, torch.Tensor):
        x_batch_cpu = x_batch.cpu().numpy()
    else:
        x_batch_cpu = x_batch  # Already a NumPy array

    # Copy and apply threshold
    a_copied_matrix = np.copy(a_batch)
    a_copied_matrix[a_copied_matrix <= threshold] = 0  # Set pixels below threshold to 0
    a_mask = (a_copied_matrix != 0)
    
    # Repeat mask along the appropriate axis
    a_reshaped_mask = np.repeat(a_mask, x_batch_cpu.shape[1], axis=1)
    a_masked_x_batch = np.copy(x_batch_cpu)
    a_masked_x_batch[~a_reshaped_mask] = 1

    # Calculate total and removed pixels
    total_pixels = np.prod(x_batch_cpu.shape)
    removed_pixels = np.sum(~a_reshaped_mask)

    # Convert back to torch tensor if needed
    if isinstance(x_batch, torch.Tensor):
        a_masked_x_batch = torch.from_numpy(a_masked_x_batch).to(x_batch.device)

    return a_masked_x_batch, removed_pixels / total_pixels

# Load and play an audio file from a specified link.
def beep(link):
    audio_file = link # Load the audio file and get the audio data and sampling rate
    audio_data, sampling_rate = librosa.load(audio_file, sr=None)
    display(Audio(data=audio_data, rate=sampling_rate, autoplay=True)) # Display the audio

"""## Probabilities"""

# Get probability distributions over classes for a batch of images using a given model.
def get_probabilities(batch, model):
  probabilities = [0] * len(batch)

  for i in range(len(batch)):
    input_image = Image.fromarray(normalize_image(batch[i]))  # Assuming the pixel values are in the range [0, 1]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch) #
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities[i] = torch.nn.functional.softmax(output[0], dim=0)
  return probabilities

"""## Results"""

# Generate a DataFrame with the top 5 predictions and probabilities for each image in the batch.
def get_results5(batch, y_batch, model):
  probabilities = get_probabilities(batch, model)

  results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                     'Class_1', 'Probability_1',
                                     'Class_2', 'Probability_2',
                                     'Class_3', 'Probability_3',
                                     'Class_4', 'Probability_4',
                                     'Class_5', 'Probability_5',])

  for i in range(len(batch)):
      top5_prob, top5_catid = torch.topk(probabilities[i], 5)
      class_list = [0] * top5_prob.size(0)
      prob_list = [0] * top5_prob.size(0)
      for j in range(top5_prob.size(0)):
        class_list[j] = categories[top5_catid[j]]
        prob_list[j] = top5_prob[j].item()

      results_df = results_df._append({'Image': i + 1,
                                      'Real': y_batch[i],
                                      'Name': CLASSES[y_batch[i]],
                                      'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                      'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                      'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                      'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                      'Class_5': class_list[4], 'Probability_5': prob_list[4]}, ignore_index=True)
  clear_output()
  return results_df

# Improved version of `get_results5` that handles both CUDA tensors and missing labels in CLASSES.
def new_get_results5(batch, y_batch, model):
    probabilities = get_probabilities(batch, model)

    results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                       'Class_1', 'Probability_1',
                                       'Class_2', 'Probability_2',
                                       'Class_3', 'Probability_3',
                                       'Class_4', 'Probability_4',
                                       'Class_5', 'Probability_5'])

    for i in range(len(batch)):
        # Get top 5 probabilities and category IDs
        top5_prob, top5_catid = torch.topk(probabilities[i], 5)
        class_list = [0] * top5_prob.size(0)
        prob_list = [0] * top5_prob.size(0)

        for j in range(top5_prob.size(0)):
            class_list[j] = categories[top5_catid[j]]
            prob_list[j] = top5_prob[j].item()

        # Convert y_batch[i] to an integer label if it's still a CUDA tensor
        real_label = int(y_batch[i].cpu().item()) if isinstance(y_batch[i], torch.Tensor) else y_batch[i]

        # Check if real_label exists in CLASSES
        if real_label in CLASSES:
            real_name = CLASSES[real_label]
        else:
            real_name = "Unknown"  # Fallback if label not found in CLASSES

        # Append row to the results DataFrame
        results_df = results_df._append({'Image': i + 1,
                                         'Real': real_label,
                                         'Name': real_name,
                                         'Class_1': class_list[0], 'Probability_1': prob_list[0],
                                         'Class_2': class_list[1], 'Probability_2': prob_list[1],
                                         'Class_3': class_list[2], 'Probability_3': prob_list[2],
                                         'Class_4': class_list[3], 'Probability_4': prob_list[3],
                                         'Class_5': class_list[4], 'Probability_5': prob_list[4]}, 
                                         ignore_index=True)

    clear_output()
    return results_df

# Generate a DataFrame with the top 1 prediction and probability for each image in the batch.
def get_results1(batch, y_batch, model):
  probabilities = get_probabilities(batch, model)

  results_df = pd.DataFrame(columns=['Image', 'Real', 'Name',
                                     'Class_1', 'Probability_1'])

  for i in range(len(batch)):
      top1_prob, top1_catid = torch.topk(probabilities[i], 1)
      class_list = [0] * top1_prob.size(0)
      prob_list = [0] * top1_prob.size(0)
      for j in range(top1_prob.size(0)):
        class_list[j] = categories[top1_catid[j]]
        prob_list[j] = top1_prob[j].item()

      results_df = results_df._append({'Image': i + 1,
                                      'Real': y_batch[i],
                                      'Name': CLASSES[y_batch[i]],
                                      'Class_1': class_list[0], 'Probability_1': prob_list[0]}, ignore_index=True)
  clear_output()
  return results_df

"""## Accuracies"""

# Checks if the actual label of an image is within the top-k predictions for that image.
def is_classified_correct(row, k):
    name_labels = [label.strip() for label in row['Name'].split(',')]

    for i in range(1, k + 1):
        class_label = row[f'Class_{i}']
        if any(label in class_label for label in name_labels):
            return True
    return False

# Calculates and displays the proportion of correctly classified images based on top-k accuracy.
def get_corrects(df, k):
  df['Match'] = df.apply(is_classified_correct, k=k, axis=1)
  print(f"Correctly classified instances: {(df['Match'].sum() / len(df))}")

"""## Main"""

# Generates classification results for the batch and retrieves top-k predictions accuracy.
def get_results(model, a_masked_x_batch, y_batch, k):
  if k == 1:
    df = get_results1(a_masked_x_batch, y_batch, model)
  elif k == 5:
    df = new_get_results5(a_masked_x_batch, y_batch, model)
  else:
    print("Error")
    return 0
  return df

"""## Heatmap"""

def create_heatmap(a_matrix, threshold, image_index):
    # Move to CPU and convert to NumPy if a_matrix is a tensor
    if isinstance(a_matrix, torch.Tensor):
        a_matrix = a_matrix.cpu().numpy()
    
    # Reduce 3D to 2D by averaging across the color channels
    if a_matrix.ndim == 3:
        a_matrix = a_matrix.mean(axis=0)

    # Normalize the importance matrix and apply threshold if needed
    plt.figure(figsize=(8, 8))
    sns.heatmap(a_matrix, cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Pixel Importance - Image {image_index + 1}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()

def create_heatmap(a_matrix, threshold, image_index):
    # Move to CPU and convert to NumPy if a_matrix is a tensor
    if isinstance(a_matrix, torch.Tensor):
        a_matrix = a_matrix.cpu().numpy()

    importance_matrix = np.where(a_matrix > threshold, a_matrix, 0)
    importance_matrix = importance_matrix / np.max(importance_matrix)

    plt.figure(figsize=(8, 8))
    sns.heatmap(importance_matrix[0], cmap='viridis', cbar=True)
    plt.title(f'Heatmap of Pixel Importance - Image {image_index + 1}')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()

"""## Catagories"""

# Define the URL where the ImageNet labels are stored
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

# Download the ImageNet labels file from the URL and save it locally
response = requests.get(url)
if response.status_code == 200:
    with open("imagenet_classes.txt", "w") as f:
        f.write(response.text) 
else:
    print("Failed to download the ImageNet labels file.") 

# Open the downloaded file and read the categories line by line
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

"""## Classes"""

CLASSES = {0: 'tench, Tinca tinca',
 1: 'goldfish, Carassius auratus',
 2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 3: 'tiger shark, Galeocerdo cuvieri',
 4: 'hammerhead, hammerhead shark',
 5: 'electric ray, crampfish, numbfish, torpedo',
 6: 'stingray',
 7: 'cock',
 8: 'hen',
 9: 'ostrich, Struthio camelus',
 10: 'brambling, Fringilla montifringilla',
 11: 'goldfinch, Carduelis carduelis',
 12: 'house finch, linnet, Carpodacus mexicanus',
 13: 'junco, snowbird',
 14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
 15: 'robin, American robin, Turdus migratorius',
 16: 'bulbul',
 17: 'jay',
 18: 'magpie',
 19: 'chickadee',
 20: 'water ouzel, dipper',
 21: 'kite',
 22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
 23: 'vulture',
 24: 'great grey owl, great gray owl, Strix nebulosa',
 25: 'European fire salamander, Salamandra salamandra',
 26: 'common newt, Triturus vulgaris',
 27: 'eft',
 28: 'spotted salamander, Ambystoma maculatum',
 29: 'axolotl, mud puppy, Ambystoma mexicanum',
 30: 'bullfrog, Rana catesbeiana',
 31: 'tree frog, tree-frog',
 32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
 33: 'loggerhead, loggerhead turtle, Caretta caretta',
 34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
 35: 'mud turtle',
 36: 'terrapin',
 37: 'box turtle, box tortoise',
 38: 'banded gecko',
 39: 'common iguana, iguana, Iguana iguana',
 40: 'American chameleon, anole, Anolis carolinensis',
 41: 'whiptail, whiptail lizard',
 42: 'agama',
 43: 'frilled lizard, Chlamydosaurus kingi',
 44: 'alligator lizard',
 45: 'Gila monster, Heloderma suspectum',
 46: 'green lizard, Lacerta viridis',
 47: 'African chameleon, Chamaeleo chamaeleon',
 48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
 49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
 50: 'American alligator, Alligator mississipiensis',
 51: 'triceratops',
 52: 'thunder snake, worm snake, Carphophis amoenus',
 53: 'ringneck snake, ring-necked snake, ring snake',
 54: 'hognose snake, puff adder, sand viper',
 55: 'green snake, grass snake',
 56: 'king snake, kingsnake',
 57: 'garter snake, grass snake',
 58: 'water snake',
 59: 'vine snake',
 60: 'night snake, Hypsiglena torquata',
 61: 'boa constrictor, Constrictor constrictor',
 62: 'rock python, rock snake, Python sebae',
 63: 'Indian cobra, Naja naja',
 64: 'green mamba',
 65: 'sea snake',
 66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
 67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
 68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
 69: 'trilobite',
 70: 'harvestman, daddy longlegs, Phalangium opilio',
 71: 'scorpion',
 72: 'black and gold garden spider, Argiope aurantia',
 73: 'barn spider, Araneus cavaticus',
 74: 'garden spider, Aranea diademata',
 75: 'black widow, Latrodectus mactans',
 76: 'tarantula',
 77: 'wolf spider, hunting spider',
 78: 'tick',
 79: 'centipede',
 80: 'black grouse',
 81: 'ptarmigan',
 82: 'ruffed grouse, partridge, Bonasa umbellus',
 83: 'prairie chicken, prairie grouse, prairie fowl',
 84: 'peacock',
 85: 'quail',
 86: 'partridge',
 87: 'African grey, African gray, Psittacus erithacus',
 88: 'macaw',
 89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
 90: 'lorikeet',
 91: 'coucal',
 92: 'bee eater',
 93: 'hornbill',
 94: 'hummingbird',
 95: 'jacamar',
 96: 'toucan',
 97: 'drake',
 98: 'red-breasted merganser, Mergus serrator',
 99: 'goose',
 100: 'black swan, Cygnus atratus',
 101: 'tusker',
 102: 'echidna, spiny anteater, anteater',
 103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
 104: 'wallaby, brush kangaroo',
 105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
 106: 'wombat',
 107: 'jellyfish',
 108: 'sea anemone, anemone',
 109: 'brain coral',
 110: 'flatworm, platyhelminth',
 111: 'nematode, nematode worm, roundworm',
 112: 'conch',
 113: 'snail',
 114: 'slug',
 115: 'sea slug, nudibranch',
 116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
 117: 'chambered nautilus, pearly nautilus, nautilus',
 118: 'Dungeness crab, Cancer magister',
 119: 'rock crab, Cancer irroratus',
 120: 'fiddler crab',
 121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
 122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
 123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
 124: 'crayfish, crawfish, crawdad, crawdaddy',
 125: 'hermit crab',
 126: 'isopod',
 127: 'white stork, Ciconia ciconia',
 128: 'black stork, Ciconia nigra',
 129: 'spoonbill',
 130: 'flamingo',
 131: 'little blue heron, Egretta caerulea',
 132: 'American egret, great white heron, Egretta albus',
 133: 'bittern',
 134: 'crane',
 135: 'limpkin, Aramus pictus',
 136: 'European gallinule, Porphyrio porphyrio',
 137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
 138: 'bustard',
 139: 'ruddy turnstone, Arenaria interpres',
 140: 'red-backed sandpiper, dunlin, Erolia alpina',
 141: 'redshank, Tringa totanus',
 142: 'dowitcher',
 143: 'oystercatcher, oyster catcher',
 144: 'pelican',
 145: 'king penguin, Aptenodytes patagonica',
 146: 'albatross, mollymawk',
 147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
 148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
 149: 'dugong, Dugong dugon',
 150: 'sea lion',
 151: 'Chihuahua',
 152: 'Japanese spaniel',
 153: 'Maltese dog, Maltese terrier, Maltese',
 154: 'Pekinese, Pekingese, Peke',
 155: 'Shih-Tzu',
 156: 'Blenheim spaniel',
 157: 'papillon',
 158: 'toy terrier',
 159: 'Rhodesian ridgeback',
 160: 'Afghan hound, Afghan',
 161: 'basset, basset hound',
 162: 'beagle',
 163: 'bloodhound, sleuthhound',
 164: 'bluetick',
 165: 'black-and-tan coonhound',
 166: 'Walker hound, Walker foxhound',
 167: 'English foxhound',
 168: 'redbone',
 169: 'borzoi, Russian wolfhound',
 170: 'Irish wolfhound',
 171: 'Italian greyhound',
 172: 'whippet',
 173: 'Ibizan hound, Ibizan Podenco',
 174: 'Norwegian elkhound, elkhound',
 175: 'otterhound, otter hound',
 176: 'Saluki, gazelle hound',
 177: 'Scottish deerhound, deerhound',
 178: 'Weimaraner',
 179: 'Staffordshire bullterrier, Staffordshire bull terrier',
 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 181: 'Bedlington terrier',
 182: 'Border terrier',
 183: 'Kerry blue terrier',
 184: 'Irish terrier',
 185: 'Norfolk terrier',
 186: 'Norwich terrier',
 187: 'Yorkshire terrier',
 188: 'wire-haired fox terrier',
 189: 'Lakeland terrier',
 190: 'Sealyham terrier, Sealyham',
 191: 'Airedale, Airedale terrier',
 192: 'cairn, cairn terrier',
 193: 'Australian terrier',
 194: 'Dandie Dinmont, Dandie Dinmont terrier',
 195: 'Boston bull, Boston terrier',
 196: 'miniature schnauzer',
 197: 'giant schnauzer',
 198: 'standard schnauzer',
 199: 'Scotch terrier, Scottish terrier, Scottie',
 200: 'Tibetan terrier, chrysanthemum dog',
 201: 'silky terrier, Sydney silky',
 202: 'soft-coated wheaten terrier',
 203: 'West Highland white terrier',
 204: 'Lhasa, Lhasa apso',
 205: 'flat-coated retriever',
 206: 'curly-coated retriever',
 207: 'golden retriever',
 208: 'Labrador retriever',
 209: 'Chesapeake Bay retriever',
 210: 'German short-haired pointer',
 211: 'vizsla, Hungarian pointer',
 212: 'English setter',
 213: 'Irish setter, red setter',
 214: 'Gordon setter',
 215: 'Brittany spaniel',
 216: 'clumber, clumber spaniel',
 217: 'English springer, English springer spaniel',
 218: 'Welsh springer spaniel',
 219: 'cocker spaniel, English cocker spaniel, cocker',
 220: 'Sussex spaniel',
 221: 'Irish water spaniel',
 222: 'kuvasz',
 223: 'schipperke',
 224: 'groenendael',
 225: 'malinois',
 226: 'briard',
 227: 'kelpie',
 228: 'komondor',
 229: 'Old English sheepdog, bobtail',
 230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
 231: 'collie',
 232: 'Border collie',
 233: 'Bouvier des Flandres, Bouviers des Flandres',
 234: 'Rottweiler',
 235: 'German shepherd, German shepherd dog, German police dog, alsatian',
 236: 'Doberman, Doberman pinscher',
 237: 'miniature pinscher',
 238: 'Greater Swiss Mountain dog',
 239: 'Bernese mountain dog',
 240: 'Appenzeller',
 241: 'EntleBucher',
 242: 'boxer',
 243: 'bull mastiff',
 244: 'Tibetan mastiff',
 245: 'French bulldog',
 246: 'Great Dane',
 247: 'Saint Bernard, St Bernard',
 248: 'Eskimo dog, husky',
 249: 'malamute, malemute, Alaskan malamute',
 250: 'Siberian husky',
 251: 'dalmatian, coach dog, carriage dog',
 252: 'affenpinscher, monkey pinscher, monkey dog',
 253: 'basenji',
 254: 'pug, pug-dog',
 255: 'Leonberg',
 256: 'Newfoundland, Newfoundland dog',
 257: 'Great Pyrenees',
 258: 'Samoyed, Samoyede',
 259: 'Pomeranian',
 260: 'chow, chow chow',
 261: 'keeshond',
 262: 'Brabancon griffon',
 263: 'Pembroke, Pembroke Welsh corgi',
 264: 'Cardigan, Cardigan Welsh corgi',
 265: 'toy poodle',
 266: 'miniature poodle',
 267: 'standard poodle',
 268: 'Mexican hairless',
 269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
 270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
 271: 'red wolf, maned wolf, Canis rufus, Canis niger',
 272: 'coyote, prairie wolf, brush wolf, Canis latrans',
 273: 'dingo, warrigal, warragal, Canis dingo',
 274: 'dhole, Cuon alpinus',
 275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
 276: 'hyena, hyaena',
 277: 'red fox, Vulpes vulpes',
 278: 'kit fox, Vulpes macrotis',
 279: 'Arctic fox, white fox, Alopex lagopus',
 280: 'grey fox, gray fox, Urocyon cinereoargenteus',
 281: 'tabby, tabby cat',
 282: 'tiger cat',
 283: 'Persian cat',
 284: 'Siamese cat, Siamese',
 285: 'Egyptian cat',
 286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
 287: 'lynx, catamount',
 288: 'leopard, Panthera pardus',
 289: 'snow leopard, ounce, Panthera uncia',
 290: 'jaguar, panther, Panthera onca, Felis onca',
 291: 'lion, king of beasts, Panthera leo',
 292: 'tiger, Panthera tigris',
 293: 'cheetah, chetah, Acinonyx jubatus',
 294: 'brown bear, bruin, Ursus arctos',
 295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
 296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
 297: 'sloth bear, Melursus ursinus, Ursus ursinus',
 298: 'mongoose',
 299: 'meerkat, mierkat',
 300: 'tiger beetle',
 301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
 302: 'ground beetle, carabid beetle',
 303: 'long-horned beetle, longicorn, longicorn beetle',
 304: 'leaf beetle, chrysomelid',
 305: 'dung beetle',
 306: 'rhinoceros beetle',
 307: 'weevil',
 308: 'fly',
 309: 'bee',
 310: 'ant, emmet, pismire',
 311: 'grasshopper, hopper',
 312: 'cricket',
 313: 'walking stick, walkingstick, stick insect',
 314: 'cockroach, roach',
 315: 'mantis, mantid',
 316: 'cicada, cicala',
 317: 'leafhopper',
 318: 'lacewing, lacewing fly',
 319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
 320: 'damselfly',
 321: 'admiral',
 322: 'ringlet, ringlet butterfly',
 323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
 324: 'cabbage butterfly',
 325: 'sulphur butterfly, sulfur butterfly',
 326: 'lycaenid, lycaenid butterfly',
 327: 'starfish, sea star',
 328: 'sea urchin',
 329: 'sea cucumber, holothurian',
 330: 'wood rabbit, cottontail, cottontail rabbit',
 331: 'hare',
 332: 'Angora, Angora rabbit',
 333: 'hamster',
 334: 'porcupine, hedgehog',
 335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
 336: 'marmot',
 337: 'beaver',
 338: 'guinea pig, Cavia cobaya',
 339: 'sorrel',
 340: 'zebra',
 341: 'hog, pig, grunter, squealer, Sus scrofa',
 342: 'wild boar, boar, Sus scrofa',
 343: 'warthog',
 344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
 345: 'ox',
 346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
 347: 'bison',
 348: 'ram, tup',
 349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
 350: 'ibex, Capra ibex',
 351: 'hartebeest',
 352: 'impala, Aepyceros melampus',
 353: 'gazelle',
 354: 'Arabian camel, dromedary, Camelus dromedarius',
 355: 'llama',
 356: 'weasel',
 357: 'mink',
 358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
 359: 'black-footed ferret, ferret, Mustela nigripes',
 360: 'otter',
 361: 'skunk, polecat, wood pussy',
 362: 'badger',
 363: 'armadillo',
 364: 'three-toed sloth, ai, Bradypus tridactylus',
 365: 'orangutan, orang, orangutang, Pongo pygmaeus',
 366: 'gorilla, Gorilla gorilla',
 367: 'chimpanzee, chimp, Pan troglodytes',
 368: 'gibbon, Hylobates lar',
 369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
 370: 'guenon, guenon monkey',
 371: 'patas, hussar monkey, Erythrocebus patas',
 372: 'baboon',
 373: 'macaque',
 374: 'langur',
 375: 'colobus, colobus monkey',
 376: 'proboscis monkey, Nasalis larvatus',
 377: 'marmoset',
 378: 'capuchin, ringtail, Cebus capucinus',
 379: 'howler monkey, howler',
 380: 'titi, titi monkey',
 381: 'spider monkey, Ateles geoffroyi',
 382: 'squirrel monkey, Saimiri sciureus',
 383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
 384: 'indri, indris, Indri indri, Indri brevicaudatus',
 385: 'Indian elephant, Elephas maximus',
 386: 'African elephant, Loxodonta africana',
 387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
 388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
 389: 'barracouta, snoek',
 390: 'eel',
 391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
 392: 'rock beauty, Holocanthus tricolor',
 393: 'anemone fish',
 394: 'sturgeon',
 395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
 396: 'lionfish',
 397: 'puffer, pufferfish, blowfish, globefish',
 398: 'abacus',
 399: 'abaya',
 400: "academic gown, academic robe, judge's robe",
 401: 'accordion, piano accordion, squeeze box',
 402: 'acoustic guitar',
 403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
 404: 'airliner',
 405: 'airship, dirigible',
 406: 'altar',
 407: 'ambulance',
 408: 'amphibian, amphibious vehicle',
 409: 'analog clock',
 410: 'apiary, bee house',
 411: 'apron',
 412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
 413: 'assault rifle, assault gun',
 414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
 415: 'bakery, bakeshop, bakehouse',
 416: 'balance beam, beam',
 417: 'balloon',
 418: 'ballpoint, ballpoint pen, ballpen, Biro',
 419: 'Band Aid',
 420: 'banjo',
 421: 'bannister, banister, balustrade, balusters, handrail',
 422: 'barbell',
 423: 'barber chair',
 424: 'barbershop',
 425: 'barn',
 426: 'barometer',
 427: 'barrel, cask',
 428: 'barrow, garden cart, lawn cart, wheelbarrow',
 429: 'baseball',
 430: 'basketball',
 431: 'bassinet',
 432: 'bassoon',
 433: 'bathing cap, swimming cap',
 434: 'bath towel',
 435: 'bathtub, bathing tub, bath, tub',
 436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
 437: 'beacon, lighthouse, beacon light, pharos',
 438: 'beaker',
 439: 'bearskin, busby, shako',
 440: 'beer bottle',
 441: 'beer glass',
 442: 'bell cote, bell cot',
 443: 'bib',
 444: 'bicycle-built-for-two, tandem bicycle, tandem',
 445: 'bikini, two-piece',
 446: 'binder, ring-binder',
 447: 'binoculars, field glasses, opera glasses',
 448: 'birdhouse',
 449: 'boathouse',
 450: 'bobsled, bobsleigh, bob',
 451: 'bolo tie, bolo, bola tie, bola',
 452: 'bonnet, poke bonnet',
 453: 'bookcase',
 454: 'bookshop, bookstore, bookstall',
 455: 'bottlecap',
 456: 'bow',
 457: 'bow tie, bow-tie, bowtie',
 458: 'brass, memorial tablet, plaque',
 459: 'brassiere, bra, bandeau',
 460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
 461: 'breastplate, aegis, egis',
 462: 'broom',
 463: 'bucket, pail',
 464: 'buckle',
 465: 'bulletproof vest',
 466: 'bullet train, bullet',
 467: 'butcher shop, meat market',
 468: 'cab, hack, taxi, taxicab',
 469: 'caldron, cauldron',
 470: 'candle, taper, wax light',
 471: 'cannon',
 472: 'canoe',
 473: 'can opener, tin opener',
 474: 'cardigan',
 475: 'car mirror',
 476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
 477: "carpenter's kit, tool kit",
 478: 'carton',
 479: 'car wheel',
 480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
 481: 'cassette',
 482: 'cassette player',
 483: 'castle',
 484: 'catamaran',
 485: 'CD player',
 486: 'cello, violoncello',
 487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
 488: 'chain',
 489: 'chainlink fence',
 490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
 491: 'chain saw, chainsaw',
 492: 'chest',
 493: 'chiffonier, commode',
 494: 'chime, bell, gong',
 495: 'china cabinet, china closet',
 496: 'Christmas stocking',
 497: 'church, church building',
 498: 'cinema, movie theater, movie theatre, movie house, picture palace',
 499: 'cleaver, meat cleaver, chopper',
 500: 'cliff dwelling',
 501: 'cloak',
 502: 'clog, geta, patten, sabot',
 503: 'cocktail shaker',
 504: 'coffee mug',
 505: 'coffeepot',
 506: 'coil, spiral, volute, whorl, helix',
 507: 'combination lock',
 508: 'computer keyboard, keypad',
 509: 'confectionery, confectionary, candy store',
 510: 'container ship, containership, container vessel',
 511: 'convertible',
 512: 'corkscrew, bottle screw',
 513: 'cornet, horn, trumpet, trump',
 514: 'cowboy boot',
 515: 'cowboy hat, ten-gallon hat',
 516: 'cradle',
 517: 'crane',
 518: 'crash helmet',
 519: 'crate',
 520: 'crib, cot',
 521: 'Crock Pot',
 522: 'croquet ball',
 523: 'crutch',
 524: 'cuirass',
 525: 'dam, dike, dyke',
 526: 'desk',
 527: 'desktop computer',
 528: 'dial telephone, dial phone',
 529: 'diaper, nappy, napkin',
 530: 'digital clock',
 531: 'digital watch',
 532: 'dining table, board',
 533: 'dishrag, dishcloth',
 534: 'dishwasher, dish washer, dishwashing machine',
 535: 'disk brake, disc brake',
 536: 'dock, dockage, docking facility',
 537: 'dogsled, dog sled, dog sleigh',
 538: 'dome',
 539: 'doormat, welcome mat',
 540: 'drilling platform, offshore rig',
 541: 'drum, membranophone, tympan',
 542: 'drumstick',
 543: 'dumbbell',
 544: 'Dutch oven',
 545: 'electric fan, blower',
 546: 'electric guitar',
 547: 'electric locomotive',
 548: 'entertainment center',
 549: 'envelope',
 550: 'espresso maker',
 551: 'face powder',
 552: 'feather boa, boa',
 553: 'file, file cabinet, filing cabinet',
 554: 'fireboat',
 555: 'fire engine, fire truck',
 556: 'fire screen, fireguard',
 557: 'flagpole, flagstaff',
 558: 'flute, transverse flute',
 559: 'folding chair',
 560: 'football helmet',
 561: 'forklift',
 562: 'fountain',
 563: 'fountain pen',
 564: 'four-poster',
 565: 'freight car',
 566: 'French horn, horn',
 567: 'frying pan, frypan, skillet',
 568: 'fur coat',
 569: 'garbage truck, dustcart',
 570: 'gasmask, respirator, gas helmet',
 571: 'gas pump, gasoline pump, petrol pump, island dispenser',
 572: 'goblet',
 573: 'go-kart',
 574: 'golf ball',
 575: 'golfcart, golf cart',
 576: 'gondola',
 577: 'gong, tam-tam',
 578: 'gown',
 579: 'grand piano, grand',
 580: 'greenhouse, nursery, glasshouse',
 581: 'grille, radiator grille',
 582: 'grocery store, grocery, food market, market',
 583: 'guillotine',
 584: 'hair slide',
 585: 'hair spray',
 586: 'half track',
 587: 'hammer',
 588: 'hamper',
 589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
 590: 'hand-held computer, hand-held microcomputer',
 591: 'handkerchief, hankie, hanky, hankey',
 592: 'hard disc, hard disk, fixed disk',
 593: 'harmonica, mouth organ, harp, mouth harp',
 594: 'harp',
 595: 'harvester, reaper',
 596: 'hatchet',
 597: 'holster',
 598: 'home theater, home theatre',
 599: 'honeycomb',
 600: 'hook, claw',
 601: 'hoopskirt, crinoline',
 602: 'horizontal bar, high bar',
 603: 'horse cart, horse-cart',
 604: 'hourglass',
 605: 'iPod',
 606: 'iron, smoothing iron',
 607: "jack-o'-lantern",
 608: 'jean, blue jean, denim',
 609: 'jeep, landrover',
 610: 'jersey, T-shirt, tee shirt',
 611: 'jigsaw puzzle',
 612: 'jinrikisha, ricksha, rickshaw',
 613: 'joystick',
 614: 'kimono',
 615: 'knee pad',
 616: 'knot',
 617: 'lab coat, laboratory coat',
 618: 'ladle',
 619: 'lampshade, lamp shade',
 620: 'laptop, laptop computer',
 621: 'lawn mower, mower',
 622: 'lens cap, lens cover',
 623: 'letter opener, paper knife, paperknife',
 624: 'library',
 625: 'lifeboat',
 626: 'lighter, light, igniter, ignitor',
 627: 'limousine, limo',
 628: 'liner, ocean liner',
 629: 'lipstick, lip rouge',
 630: 'Loafer',
 631: 'lotion',
 632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
 633: "loupe, jeweler's loupe",
 634: 'lumbermill, sawmill',
 635: 'magnetic compass',
 636: 'mailbag, postbag',
 637: 'mailbox, letter box',
 638: 'maillot',
 639: 'maillot, tank suit',
 640: 'manhole cover',
 641: 'maraca',
 642: 'marimba, xylophone',
 643: 'mask',
 644: 'matchstick',
 645: 'maypole',
 646: 'maze, labyrinth',
 647: 'measuring cup',
 648: 'medicine chest, medicine cabinet',
 649: 'megalith, megalithic structure',
 650: 'microphone, mike',
 651: 'microwave, microwave oven',
 652: 'military uniform',
 653: 'milk can',
 654: 'minibus',
 655: 'miniskirt, mini',
 656: 'minivan',
 657: 'missile',
 658: 'mitten',
 659: 'mixing bowl',
 660: 'mobile home, manufactured home',
 661: 'Model T',
 662: 'modem',
 663: 'monastery',
 664: 'monitor',
 665: 'moped',
 666: 'mortar',
 667: 'mortarboard',
 668: 'mosque',
 669: 'mosquito net',
 670: 'motor scooter, scooter',
 671: 'mountain bike, all-terrain bike, off-roader',
 672: 'mountain tent',
 673: 'mouse, computer mouse',
 674: 'mousetrap',
 675: 'moving van',
 676: 'muzzle',
 677: 'nail',
 678: 'neck brace',
 679: 'necklace',
 680: 'nipple',
 681: 'notebook, notebook computer',
 682: 'obelisk',
 683: 'oboe, hautboy, hautbois',
 684: 'ocarina, sweet potato',
 685: 'odometer, hodometer, mileometer, milometer',
 686: 'oil filter',
 687: 'organ, pipe organ',
 688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
 689: 'overskirt',
 690: 'oxcart',
 691: 'oxygen mask',
 692: 'packet',
 693: 'paddle, boat paddle',
 694: 'paddlewheel, paddle wheel',
 695: 'padlock',
 696: 'paintbrush',
 697: "pajama, pyjama, pj's, jammies",
 698: 'palace',
 699: 'panpipe, pandean pipe, syrinx',
 700: 'paper towel',
 701: 'parachute, chute',
 702: 'parallel bars, bars',
 703: 'park bench',
 704: 'parking meter',
 705: 'passenger car, coach, carriage',
 706: 'patio, terrace',
 707: 'pay-phone, pay-station',
 708: 'pedestal, plinth, footstall',
 709: 'pencil box, pencil case',
 710: 'pencil sharpener',
 711: 'perfume, essence',
 712: 'Petri dish',
 713: 'photocopier',
 714: 'pick, plectrum, plectron',
 715: 'pickelhaube',
 716: 'picket fence, paling',
 717: 'pickup, pickup truck',
 718: 'pier',
 719: 'piggy bank, penny bank',
 720: 'pill bottle',
 721: 'pillow',
 722: 'ping-pong ball',
 723: 'pinwheel',
 724: 'pirate, pirate ship',
 725: 'pitcher, ewer',
 726: "plane, carpenter's plane, woodworking plane",
 727: 'planetarium',
 728: 'plastic bag',
 729: 'plate rack',
 730: 'plow, plough',
 731: "plunger, plumber's helper",
 732: 'Polaroid camera, Polaroid Land camera',
 733: 'pole',
 734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
 735: 'poncho',
 736: 'pool table, billiard table, snooker table',
 737: 'pop bottle, soda bottle',
 738: 'pot, flowerpot',
 739: "potter's wheel",
 740: 'power drill',
 741: 'prayer rug, prayer mat',
 742: 'printer',
 743: 'prison, prison house',
 744: 'projectile, missile',
 745: 'projector',
 746: 'puck, hockey puck',
 747: 'punching bag, punch bag, punching ball, punchball',
 748: 'purse',
 749: 'quill, quill pen',
 750: 'quilt, comforter, comfort, puff',
 751: 'racer, race car, racing car',
 752: 'racket, racquet',
 753: 'radiator',
 754: 'radio, wireless',
 755: 'radio telescope, radio reflector',
 756: 'rain barrel',
 757: 'recreational vehicle, RV, R.V.',
 758: 'reel',
 759: 'reflex camera',
 760: 'refrigerator, icebox',
 761: 'remote control, remote',
 762: 'restaurant, eating house, eating place, eatery',
 763: 'revolver, six-gun, six-shooter',
 764: 'rifle',
 765: 'rocking chair, rocker',
 766: 'rotisserie',
 767: 'rubber eraser, rubber, pencil eraser',
 768: 'rugby ball',
 769: 'rule, ruler',
 770: 'running shoe',
 771: 'safe',
 772: 'safety pin',
 773: 'saltshaker, salt shaker',
 774: 'sandal',
 775: 'sarong',
 776: 'sax, saxophone',
 777: 'scabbard',
 778: 'scale, weighing machine',
 779: 'school bus',
 780: 'schooner',
 781: 'scoreboard',
 782: 'screen, CRT screen',
 783: 'screw',
 784: 'screwdriver',
 785: 'seat belt, seatbelt',
 786: 'sewing machine',
 787: 'shield, buckler',
 788: 'shoe shop, shoe-shop, shoe store',
 789: 'shoji',
 790: 'shopping basket',
 791: 'shopping cart',
 792: 'shovel',
 793: 'shower cap',
 794: 'shower curtain',
 795: 'ski',
 796: 'ski mask',
 797: 'sleeping bag',
 798: 'slide rule, slipstick',
 799: 'sliding door',
 800: 'slot, one-armed bandit',
 801: 'snorkel',
 802: 'snowmobile',
 803: 'snowplow, snowplough',
 804: 'soap dispenser',
 805: 'soccer ball',
 806: 'sock',
 807: 'solar dish, solar collector, solar furnace',
 808: 'sombrero',
 809: 'soup bowl',
 810: 'space bar',
 811: 'space heater',
 812: 'space shuttle',
 813: 'spatula',
 814: 'speedboat',
 815: "spider web, spider's web",
 816: 'spindle',
 817: 'sports car, sport car',
 818: 'spotlight, spot',
 819: 'stage',
 820: 'steam locomotive',
 821: 'steel arch bridge',
 822: 'steel drum',
 823: 'stethoscope',
 824: 'stole',
 825: 'stone wall',
 826: 'stopwatch, stop watch',
 827: 'stove',
 828: 'strainer',
 829: 'streetcar, tram, tramcar, trolley, trolley car',
 830: 'stretcher',
 831: 'studio couch, day bed',
 832: 'stupa, tope',
 833: 'submarine, pigboat, sub, U-boat',
 834: 'suit, suit of clothes',
 835: 'sundial',
 836: 'sunglass',
 837: 'sunglasses, dark glasses, shades',
 838: 'sunscreen, sunblock, sun blocker',
 839: 'suspension bridge',
 840: 'swab, swob, mop',
 841: 'sweatshirt',
 842: 'swimming trunks, bathing trunks',
 843: 'swing',
 844: 'switch, electric switch, electrical switch',
 845: 'syringe',
 846: 'table lamp',
 847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
 848: 'tape player',
 849: 'teapot',
 850: 'teddy, teddy bear',
 851: 'television, television system',
 852: 'tennis ball',
 853: 'thatch, thatched roof',
 854: 'theater curtain, theatre curtain',
 855: 'thimble',
 856: 'thresher, thrasher, threshing machine',
 857: 'throne',
 858: 'tile roof',
 859: 'toaster',
 860: 'tobacco shop, tobacconist shop, tobacconist',
 861: 'toilet seat',
 862: 'torch',
 863: 'totem pole',
 864: 'tow truck, tow car, wrecker',
 865: 'toyshop',
 866: 'tractor',
 867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
 868: 'tray',
 869: 'trench coat',
 870: 'tricycle, trike, velocipede',
 871: 'trimaran',
 872: 'tripod',
 873: 'triumphal arch',
 874: 'trolleybus, trolley coach, trackless trolley',
 875: 'trombone',
 876: 'tub, vat',
 877: 'turnstile',
 878: 'typewriter keyboard',
 879: 'umbrella',
 880: 'unicycle, monocycle',
 881: 'upright, upright piano',
 882: 'vacuum, vacuum cleaner',
 883: 'vase',
 884: 'vault',
 885: 'velvet',
 886: 'vending machine',
 887: 'vestment',
 888: 'viaduct',
 889: 'violin, fiddle',
 890: 'volleyball',
 891: 'waffle iron',
 892: 'wall clock',
 893: 'wallet, billfold, notecase, pocketbook',
 894: 'wardrobe, closet, press',
 895: 'warplane, military plane',
 896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
 897: 'washer, automatic washer, washing machine',
 898: 'water bottle',
 899: 'water jug',
 900: 'water tower',
 901: 'whiskey jug',
 902: 'whistle',
 903: 'wig',
 904: 'window screen',
 905: 'window shade',
 906: 'Windsor tie',
 907: 'wine bottle',
 908: 'wing',
 909: 'wok',
 910: 'wooden spoon',
 911: 'wool, woolen, woollen',
 912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
 913: 'wreck',
 914: 'yawl',
 915: 'yurt',
 916: 'web site, website, internet site, site',
 917: 'comic book',
 918: 'crossword puzzle, crossword',
 919: 'street sign',
 920: 'traffic light, traffic signal, stoplight',
 921: 'book jacket, dust cover, dust jacket, dust wrapper',
 922: 'menu',
 923: 'plate',
 924: 'guacamole',
 925: 'consomme',
 926: 'hot pot, hotpot',
 927: 'trifle',
 928: 'ice cream, icecream',
 929: 'ice lolly, lolly, lollipop, popsicle',
 930: 'French loaf',
 931: 'bagel, beigel',
 932: 'pretzel',
 933: 'cheeseburger',
 934: 'hotdog, hot dog, red hot',
 935: 'mashed potato',
 936: 'head cabbage',
 937: 'broccoli',
 938: 'cauliflower',
 939: 'zucchini, courgette',
 940: 'spaghetti squash',
 941: 'acorn squash',
 942: 'butternut squash',
 943: 'cucumber, cuke',
 944: 'artichoke, globe artichoke',
 945: 'bell pepper',
 946: 'cardoon',
 947: 'mushroom',
 948: 'Granny Smith',
 949: 'strawberry',
 950: 'orange',
 951: 'lemon',
 952: 'fig',
 953: 'pineapple, ananas',
 954: 'banana',
 955: 'jackfruit, jak, jack',
 956: 'custard apple',
 957: 'pomegranate',
 958: 'hay',
 959: 'carbonara',
 960: 'chocolate sauce, chocolate syrup',
 961: 'dough',
 962: 'meat loaf, meatloaf',
 963: 'pizza, pizza pie',
 964: 'potpie',
 965: 'burrito',
 966: 'red wine',
 967: 'espresso',
 968: 'cup',
 969: 'eggnog',
 970: 'alp',
 971: 'bubble',
 972: 'cliff, drop, drop-off',
 973: 'coral reef',
 974: 'geyser',
 975: 'lakeside, lakeshore',
 976: 'promontory, headland, head, foreland',
 977: 'sandbar, sand bar',
 978: 'seashore, coast, seacoast, sea-coast',
 979: 'valley, vale',
 980: 'volcano',
 981: 'ballplayer, baseball player',
 982: 'groom, bridegroom',
 983: 'scuba diver',
 984: 'rapeseed',
 985: 'daisy',
 986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
 987: 'corn',
 988: 'acorn',
 989: 'hip, rose hip, rosehip',
 990: 'buckeye, horse chestnut, conker',
 991: 'coral fungus',
 992: 'agaric',
 993: 'gyromitra',
 994: 'stinkhorn, carrion fungus',
 995: 'earthstar',
 996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
 997: 'bolete',
 998: 'ear, spike, capitulum',
 999: 'toilet tissue, toilet paper, bathroom tissue'}

# Update the paths to your local file locations
x_link = "imagenet_samples/x_batch.pt"
y_link = "imagenet_samples/y_batch.pt"
s_link = "imagenet_samples/s_batch.pt"
SAVE_FILES_LINK = "data/"
BEEP_LINK = "data/beep-01a.mp3"

def main():
    print("main")

    # Load device: Check if a CUDA-compatible GPU is available and set the computation device accordingly.
    # If a GPU is available, it sets the device to 'cuda'; otherwise, it defaults to 'cpu'.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    # Set the number of samples to process
    nr_samples = 200

    # Load image batches: Loads `nr_samples` samples for x (input images), y (labels), and s (saliency maps)
    # from the specified links and moves them to the given device (CPU or GPU).
    x_batch, y_batch, s_batch = load_images(nr_samples, x_link, y_link, s_link, device)

    # Print the number of matches (samples) loaded.
    print(f"{len(x_batch)} matches found\n")

    # Set the number of top-k predictions to consider for accuracy evaluation
    k = 5

    # Set the threshold for pixel removal in the saliency mask
    threshold = 0.01

    # Define the explanation method to be used
    method = 'Saliency'

    # Load the pre-trained model `v3_small` and assign it to the device for computation
    model, device = v3_small()

    # Generate explanations: Uses the `quantus.explain` function with the selected method
    # to calculate saliency maps (`a_batch`) based on the model’s predictions for `x_batch`.
    a_batch = quantus.explain(model, x_batch, y_batch, method=method, device=device)

    # Remove pixels below the specified threshold in the explanation maps and calculate the masked x_batch.
    # `a_masked_x_batch` is the result of applying the mask, and `removed` gives the proportion of pixels removed.
    a_masked_x_batch, removed = new_remove_pixels(a_batch, x_batch, threshold)

    # Retrieve results and accuracy: Evaluate the top-k predictions based on the masked batch.
    # The function `get_results` returns a DataFrame with results.
    df = get_results(model, a_masked_x_batch, y_batch, k)  # updated v2.0

    # Print the proportion of area removed from the instances (masked pixels).
    print("Area removed from instances: " + str(removed))

    # Calculate and print classification accuracy based on the top-k matches in `df`.
    get_corrects(df, k)

    # Save the results DataFrame as a CSV file with a filename that includes the method, threshold, and k-value.
    df.to_csv(f"{SAVE_FILES_LINK}a_batch_{method}_{threshold}_{k}.csv", index=False)

    # Play a sound from the specified file link to signal completion of processing.
    #beep(BEEP_LINK)

    # Normalize the first image from the masked batch using the normalization function for saliency maps.
    normalize_image(a_masked_x_batch[0])

    # Create a dictionary mapping explanation method names to their corresponding saliency maps.
    explanations = {
        "Saliency": a_batch,
    }

    # Set the index of the image to be displayed
    index = 11

    # Create a subplot with 1 + len(explanations) columns (1 for the original image, others for explanations)
    fig, axes = plt.subplots(nrows=1, ncols=1+len(explanations), figsize=(15, 8))

    # Display the original image, denormalized, with normalization applied to the ImageNet standard.
    # The image is denormalized using the mean and std values for ImageNet, and the axis is turned off.
    axes[0].imshow(np.moveaxis(quantus.normalise_func.denormalise(x_batch[index].cpu().numpy(),
                                                              mean=np.array([0.485, 0.456, 0.406]),
                                                              std=np.array([0.229, 0.224, 0.225])), 0, -1), vmin=0.0, vmax=1.0)

    # Set the title of the first subplot (original image) to show its corresponding ImageNet class.
    axes[0].title.set_text(f"ImageNet class {y_batch[index].item()}")

    # Hide the axes of the original image
    axes[0].axis("off")

    # Loop through the explanations dictionary and display each saliency map in the subsequent subplots.
    # The saliency maps are normalized and displayed using the 'seismic' color map.
    for i, (k, v) in enumerate(explanations.items()):
        axes[i+1].imshow(quantus.normalise_func.normalise_by_negative(explanations[k][index].reshape(224, 224)), cmap="seismic", vmin=-1.0, vmax=1.0)
        axes[i+1].title.set_text(f"{k}")  # Set the title of each subplot to the explanation method name
        axes[i+1].axis("off")  # Hide the axes for the explanation maps

    # Loop through each image in the batch (`a_masked_x_batch`) by iterating over its first dimension (batch size).
    for i in range(a_masked_x_batch.shape[0]):
        # Generate and display a heatmap for the i-th image in the batch using the `create_heatmap()` function.
       create_heatmap(a_masked_x_batch[i], threshold, i)

    """## Info"""

    # Call the `exist_models()` function to check or load any existing models.
    exist_models()

    # Call the `available_methods_captum()` function from the `quantus.helpers.constants` module to list available explanation methods in Captum.
    quantus.helpers.constants.available_methods_captum()

    # Call the `available_metrics()` function from the `quantus.helpers.constants` module to list available metrics for evaluation.
    quantus.helpers.constants.available_metrics()

    """## Images"""

    # Normalize the second image from the masked batch (`a_masked_x_batch`) using the `normalize_image()` function.
    normalize_image(a_masked_x_batch[1])

    # Normalize the second image from the saliency batch (`a_batch`) using the `normalize_image()` function.
    normalize_image(a_batch[1])


if __name__ == "__main__":
    main()