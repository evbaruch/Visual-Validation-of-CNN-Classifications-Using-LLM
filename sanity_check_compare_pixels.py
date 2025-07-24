import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from PIL import Image
import pandas as pd
import numpy as np
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)
)
model.load_state_dict(torch.load("data/weights/cancer_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Accuracy results
results = []

# Base paths
xai_name = "gradcam"
original_path = "data/source/CervicalCancer/JPEG/CROPPED"
base_masked_path = f"data/mid_CervicalCancer2/{xai_name}"

# Evaluate each perturbation percentage
for pct_float in np.arange(0.05, 1.00, 0.05):
    pct = int(pct_float * 100)
    delta_expected = pct

    corrupted_folder = os.path.join(base_masked_path, f"{pct:02d}")
    corrupted_ds = datasets.ImageFolder(corrupted_folder, transform=transform)
    full_ds = datasets.ImageFolder(original_path, transform=transform)

    # assert corrupted_ds.classes == full_ds.classes, "Class mismatch!"
    # assert len(corrupted_ds) == len(full_ds), "Image count mismatch!"
    

    # Compare pixel differences
    diffs = []
    for (corrupted_path, lable_corrupted_path), (full_path, lable_full_path) in zip(corrupted_ds.samples, full_ds.samples):
        img_corrupt = Image.open(corrupted_path).convert("RGB").resize((224, 224))
        img_full = Image.open(full_path).convert("RGB").resize((224, 224))
        
        # print(f"Comparing: masked {corrupted_path} with original {full_path}")
        # print(f"Label: masked {lable_corrupted_path} vs original {lable_full_path}")
        
        if lable_corrupted_path != lable_full_path:
            print(f"Warning: Label mismatch for {corrupted_path} vs {full_path}")
            print(f"Comparing: masked {corrupted_path} with original {full_path}")
            print(f"Label: masked {lable_corrupted_path} vs original {lable_full_path}")          
            continue
        
        
        
        arr1 = np.array(img_corrupt).astype(np.int16)
        arr2 = np.array(img_full).astype(np.int16)

        delta = 1  # allowed difference per channel
        x = np.abs(arr1 - arr2)
        y = np.abs(arr1 - arr2) > delta
        diff = np.sum(np.abs(arr1 - arr2) > delta)

        total = arr1.size
        delta_pct = (diff / total) * 100
        diffs.append(delta_pct)

    avg_diff = np.mean(diffs)
    print(f"[{pct}% masked] Average pixel difference: {avg_diff:.2f}% (expected ~{delta_expected}%)")

    # # Optionally skip if mismatch is too large
    # if abs(avg_diff - delta_expected) > 2:
    #     print(f"Warning: actual diff {avg_diff:.2f}% != expected {delta_expected}% — skipping")
    #     results.append(np.nan)
    #     continue

    # Inference
    loader = torch.utils.data.DataLoader(corrupted_ds, batch_size=64, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labs).sum().item()
            total += labs.size(0)

    acc = 100 * correct / total
    print(f"Accuracy at {pct}%: {acc:.2f}%")
    results.append(acc)

# Save results to CSV (1 row: accuracy per corruption percentage)
columns = [f"{int(p * 100)}%" for p in np.arange(0.05, 1.00, 0.05)]
df = pd.DataFrame([results], index=[xai_name], columns=columns)
df.to_csv(f"data/results/CervicalCancer_eval_compare_{xai_name}.csv")
