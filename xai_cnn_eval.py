import torch, torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, datasets
from torchvision.models import resnet18
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1) recreate your architecture
model = resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)            # 5 classes
)
model.load_state_dict(torch.load("data/weights/cancer_resnet18_modified_1-2.pth", map_location=device))
model.to(device).eval()

# 2) your normal test set transform
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

results_mat = torch.zeros((19, 9))
xai_methods = ["random", "gradientshap", "gradcam", "guided_backprop","guided_gradcam","integrated_gradients","saliency","inputxgradient","smoothgrad"]

os.makedirs("data/results", exist_ok=True)

for xai_method_idx, xai_method in enumerate(xai_methods):
    for i, index in zip(np.arange(0.05, 1.00, 0.05), range(19)):
        pct = int(i * 100)
        test_ds = datasets.ImageFolder(f"data/mid_CervicalCancer_modified_1-2/{xai_method}/{pct:02d}", transform=tfm)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labs in test_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                preds = model(imgs)
                preds = preds.argmax(1)
                correct += (preds==labs).sum().item()
                total += labs.size(0)

        print(f"Baseline test accuracy: {100*correct/total:.2f}% for {xai_method} at {pct}%")
        results_mat[index, xai_method_idx] = 100 * (correct / total)

    # Save results and plot after each XAI method
    results_df = pd.DataFrame(
        results_mat[:, :xai_method_idx+1].numpy(),
        columns=xai_methods[:xai_method_idx+1],
        index=[f"{int(p*100)}%" for p in np.arange(0.05, 1.00, 0.05)]
    )
    results_df = results_df.T
    results_df.to_csv("data/results/CervicalCancer_modified_1-2.csv")

    # load the csv
results_df = pd.read_csv("data/results/CervicalCancer_modified_1-2.csv", index_col=0)


plt.figure(figsize=(10, 6))
for method in results_df.index:
    y = results_df.loc[method]
    if method == "x":
        # Only use columns 30% and above for random
        col_indices = [i for i, x in enumerate(results_df.columns) if int(x.replace('%','')) >= 30]
        x_vals = [str(int(int(results_df.columns[i].replace('%','')) - 25)) + '%' for i in col_indices]
        y_vals = y.iloc[col_indices]
        plt.plot(x_vals, y_vals, label=method, marker='o')
    else:
        plt.plot(results_df.columns, y, label=method, marker='o')
plt.xlabel("Percentage of Data Masked")
plt.ylabel("Accuracy (%)")
plt.title("XAI Methods Comparison - Cervical Cancer Dataset Top-1 Accuracy (Resnet18 with Transfer Learning)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("data/results/CervicalCancer_eval_modified_1-2.png")
plt.close()