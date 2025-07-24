# sanity_check.py

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
import pandas as pd
import numpy as np
import os

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Load model architecture
model = resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)  # 5 output classes
)
model.load_state_dict(torch.load("data/weights/cancer_resnet18_modified.pth", map_location=device))
model.to(device)
model.eval()

# 2) Define test data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3) Load or create results matrix
results_csv_path = "data/results/CervicalCancer_eval_2.csv"
if os.path.exists(results_csv_path):
    results_mat = pd.read_csv(results_csv_path, index_col=0).values
else:
    print("No previous results found, starting fresh.")
    results_mat = np.zeros((19, 9))  # 19 rows (percentages), 9 XAI methods

xai_methods = [
    "gradcam", "gradientshap", "guided_backprop", "guided_gradcam",
    "integrated_gradients", "random", "saliency", "inputxgradient", "smoothgrad"
]

model_softmax = []
actual_labels = []

# 4) Evaluate over different percentages of noise/corruption/data split
for index, pct_float in enumerate(np.arange(0.05, 1.00, 0.05)):
    pct = int(pct_float * 100)
    data_path = f"data/mid_CervicalCancer/gradcam/{pct:02d}"
    
    test_ds = datasets.ImageFolder(data_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            model_softmax.append(probs.cpu().numpy())
            actual_labels.append(labels.cpu().numpy())

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Baseline test accuracy: {accuracy:.2f}% for gradcam at {pct}%")

    # Save accuracy in 'random' column (index 5 in results_mat)
    results_mat[index, 0] = accuracy

# 5) Save predictions and labels to check.csv
actual_labels_flat = np.concatenate(actual_labels)
model_softmax_flat = np.concatenate(model_softmax)  # shape [N, num_classes]

df = pd.DataFrame(model_softmax_flat, columns=[f"softmax_{j}" for j in range(model_softmax_flat.shape[1])])
df.insert(0, "actual_label", actual_labels_flat)
df.to_csv("data/results/check.csv", index=False)

# 6) Save results matrix to CSV (transposed)
results_df = pd.DataFrame(
    results_mat,
    columns=xai_methods,
    index=[f"{int(p*100)}%" for p in np.arange(0.05, 1.00, 0.05)]
)

# Transpose: XAI methods as rows, percentages as columns
results_df = results_df.T
results_df.to_csv(results_csv_path)
