import os
import csv
import urllib.request
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ==================== MODEL CONFIGURATION ====================
# Choose your model here - easily switch between different architectures
MODEL_CONFIG = {
    "resnet18": {
        "model_func": models.resnet18,
        "weights": models.ResNet18_Weights.IMAGENET1K_V1,
        "name": "ResNet18"
    },
    "mobilenet_v2": {
        "model_func": models.mobilenet_v2,
        "weights": models.MobileNet_V2_Weights.IMAGENET1K_V1,
        "name": "MobileNetV2"
    },
    "resnet50": {
        "model_func": models.resnet50,
        "weights": models.ResNet50_Weights.IMAGENET1K_V1,
        "name": "ResNet50"
    },
    "efficientnet_b0": {
        "model_func": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "name": "EfficientNetB0"
    },
    "vgg16": {
        "model_func": models.vgg16,
        "weights": models.VGG16_Weights.IMAGENET1K_V1,
        "name": "VGG16"
    }
}

# Select model here - change this to switch models
SELECTED_MODEL = "mobilenet_v2"  # Change to "resnet18", "resnet50", "efficientnet_b0", "vgg16", etc.

# Get model configuration
if SELECTED_MODEL not in MODEL_CONFIG:
    raise ValueError(f"Model '{SELECTED_MODEL}' not found. Available: {list(MODEL_CONFIG.keys())}")

model_config = MODEL_CONFIG[SELECTED_MODEL]
MODEL_NAME = model_config["name"]
print(f"Using model: {MODEL_NAME}")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Disable cuDNN benchmarking for consistent results
torch.backends.cudnn.benchmark = False

# Load selected model
model = model_config["model_func"](weights=model_config["weights"]).to(device)
model.eval()

# ImageNet label map
with urllib.request.urlopen("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt") as url:
    idx_to_label = [line.decode("utf-8").strip() for line in url]

# Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Base directories
base_dirs = {
    "random": r"C:\imagenet_results\random",
    "saliency": r"C:\imagenet_results\saliency",
    "guided_gradcam": r"C:\imagenet_results\guided_gradcam",
    "gradientshap": r"C:\imagenet_results\gradientshap",
    "inputxgradient": r"C:\imagenet_results\inputxgradient",
    "gradcam": r"C:\imagenet_results\gradcam",
    "smoothgrad": r"C:\imagenet_results\smoothgrad",
    "integrated_gradients": r"C:\imagenet_results\integrated_gradients",
    "guided_backprop": r"C:\imagenet_results\guided_backprop                                                                                                                                                ",
}

# Percentages (excluding 100)
percentages = list(range(5, 100, 5))
valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

results_top1 = {"Percentage": percentages}
results_top5 = {"Percentage": percentages}
for method in base_dirs:
    results_top1[method] = []
    results_top5[method] = []


# Helper to preprocess image and get label
def load_image_and_class(img_path, method_suffixes, method, folder):
    try:
        # Get class name
        root = os.path.dirname(img_path)
        file = os.path.basename(img_path)
        if root != folder:
            class_name = os.path.basename(root).lower().replace("_", " ")
        else:
            filename_parts = file.split('_')
            if len(filename_parts) >= 2 and filename_parts[0].startswith('n'):
                class_part = '_'.join(filename_parts[1:]).rsplit('.', 1)[0]
                for suffix in method_suffixes:
                    if class_part.endswith('_' + suffix):
                        class_name = class_part[:-len('_' + suffix)].lower().replace("_", " ")
                        break
                else:
                    class_name = class_part.lower().replace("_", " ")
            else:
                class_name = os.path.splitext(file)[0].lower().replace("_", " ")

        image = Image.open(img_path).convert("RGB")
        tensor = transform(image)
        return tensor, class_name, file
    except Exception as e:
        return None


# Build suffix list
method_suffixes = [f"{method}{pct:02}" for method in base_dirs for pct in percentages]

# Evaluate each method
for method, base_path in base_dirs.items():
    print(f"\n=== Evaluating {method} ===")
    for pct in percentages:
        folder = os.path.join(base_path, f"{method}{pct:02}")
        if not os.path.exists(folder):
            continue

        # Gather valid files
        valid_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_extensions:
                    valid_files.append(os.path.join(root, f))

        if not valid_files:
            print(f"[SKIP] No valid images in {folder}")
            results_top1[method].append(None)
            results_top5[method].append(None)
            continue

        # OPTIMIZATION 1: Increase batch size significantly
        batch_size = 128  # Increased from 32 to better utilize GPU

        # OPTIMIZATION 2: Reduce CPU thread count to avoid bottleneck
        max_workers = min(8, os.cpu_count())  # Limit CPU threads

        # Load images in parallel (CPU threads) with limited workers
        images, class_names, file_names = [], [], []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(lambda f: load_image_and_class(f, method_suffixes, method, folder), valid_files),
                     total=len(valid_files), desc=f"{method}{pct:02} loading"))
        for res in results:
            if res:
                tensor, class_name, file = res
                images.append(tensor)
                class_names.append(class_name)
                file_names.append(file)

        # OPTIMIZATION 3: Process larger batches with GPU warming
        top1_correct, top5_correct, total = 0, 0, 0

        # Warm up GPU with a dummy batch
        if len(images) > 0:
            dummy_batch = torch.stack([images[0]] * min(batch_size, len(images))).to(device)
            with torch.no_grad():
                _ = model(dummy_batch)

        # OPTIMIZATION 4: Use non_blocking transfer and pin memory
        for i in range(0, len(images), batch_size):
            batch_tensors = torch.stack(images[i:i + batch_size])
            # Pin memory for faster GPU transfer
            batch_tensors = batch_tensors.pin_memory().to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(batch_tensors)
                _, top1 = outputs.topk(1, dim=1)
                _, top5 = outputs.topk(5, dim=1)

            # Move results back to CPU for processing
            top1_cpu = top1.cpu()
            top5_cpu = top5.cpu()

            for j in range(batch_tensors.size(0)):
                class_name = class_names[i + j]
                pred_label_top1 = idx_to_label[top1_cpu[j].item()].lower()
                pred_labels_top5 = [idx_to_label[idx].lower() for idx in top5_cpu[j]]

                variants = [class_name,
                            class_name.replace("_", " "),
                            class_name.replace(" ", "_")]

                match_top1 = any(variant in pred_label_top1 for variant in variants)
                match_top5 = any(any(variant in lbl for variant in variants) for lbl in pred_labels_top5)

                if match_top1:
                    top1_correct += 1
                if match_top5:
                    top5_correct += 1
                total += 1

        acc1 = top1_correct / total if total > 0 else None
        acc5 = top5_correct / total if total > 0 else None
        results_top1[method].append(acc1)
        results_top5[method].append(acc5)
        print(f"{method}{pct:02}: Top-1 = {acc1:.3f}, Top-5 = {acc5:.3f} ({total} images)")

# Save to CSV in the organized format
csv_file = f"results_{SELECTED_MODEL.lower()}.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)

    # Write the main header
    writer.writerow([f"Performance Results by Method and Metric - {MODEL_NAME}"])
    writer.writerow([])

    # Write Top1 Results section
    writer.writerow(["Top1 Results"])

    # Write Top1 header row
    header = ["Method"] + percentages
    writer.writerow(header)

    # Write Top1 data rows
    for method in base_dirs:
        row = [method] + [
            results_top1[method][i] if i < len(results_top1[method]) and results_top1[method][i] is not None else ""
            for i in range(len(percentages))]
        writer.writerow(row)

    writer.writerow([])  # Empty row separator

    # Write Top5 Results section
    writer.writerow(["Top5 Results"])

    # Write Top5 header row
    writer.writerow(header)

    # Write Top5 data rows
    for method in base_dirs:
        row = [method] + [
            results_top5[method][i] if i < len(results_top5[method]) and results_top5[method][i] is not None else ""
            for i in range(len(percentages))]
        writer.writerow(row)

print(f"Results saved to: {csv_file}")

# Plot Top-1 Accuracy
plt.figure(figsize=(12, 8))
for method in base_dirs:
    plt.plot(results_top1["Percentage"], results_top1[method], marker='o', label=method, linewidth=2)
plt.xlabel("Percentage of Pixels Removed", fontsize=12)
plt.ylabel("Top-1 Accuracy", fontsize=12)
plt.title(f"XAI Method Comparison - Top-1 Accuracy ({MODEL_NAME})", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_filename_top1 = f"{SELECTED_MODEL.lower()}_top1_accuracy.png"
plt.savefig(plot_filename_top1, dpi=300, bbox_inches='tight')
plt.show()

# Plot Top-5 Accuracy
plt.figure(figsize=(12, 8))
for method in base_dirs:
    plt.plot(results_top5["Percentage"], results_top5[method], marker='s', label=method, linewidth=2)
plt.xlabel("Percentage of Pixels Removed", fontsize=12)
plt.ylabel("Top-5 Accuracy", fontsize=12)
plt.title(f"XAI Method Comparison - Top-5 Accuracy ({MODEL_NAME})", fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_filename_top5 = f"{SELECTED_MODEL.lower()}_top5_accuracy.png"
plt.savefig(plot_filename_top5, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plots saved as: {plot_filename_top1} and {plot_filename_top5}")
print(f"Evaluation complete for {MODEL_NAME}!")