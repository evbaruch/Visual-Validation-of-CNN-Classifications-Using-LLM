# xai_eval_with_config.py
import os, sys, json, csv
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# 1) Load config
if len(sys.argv) != 2:
    print("Usage: python xai_eval_with_config.py <config.json>")
    sys.exit(1)
cfg = json.load(open(sys.argv[1]))

INPUT_FOLDER   = cfg["mid_folder"]
RESULTS_CSV    = cfg.get("results_csv", f"results_{cfg['model_name']}.csv")
MODEL_NAME     = cfg["model_name"]
MODEL_WEIGHTS  = cfg.get("model_weights")   # "DEFAULT" or null
WEIGHTS_PATH   = cfg.get("weights_path")    # path to .pth or null
NUM_CLASSES    = cfg["num_classes"]
IMG_SIZE       = cfg["img_size"]
MEAN           = cfg["normalize_mean"]
STD            = cfg["normalize_std"]
THRESHOLDS     = cfg["thresholds"]
PCTS           = [int(t*100) for t in THRESHOLDS]
LABEL_CFG      = cfg["label_extraction"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, Model: {MODEL_NAME}")

# 2) Build & load model
def build_model():
    if WEIGHTS_PATH:
        m = models.resnet18(pretrained=False)
        state = torch.load(WEIGHTS_PATH, map_location=device)
        # Rebuild the exact fc head you trained
        if any(k.startswith("fc.0.") for k in state):
            hidden = state["fc.0.weight"].shape[0]
            outdim = state["fc.3.bias"].shape[0]
            m.fc = nn.Sequential(
                nn.Linear(m.fc.in_features, hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden, outdim),
            )
        else:
            outdim = state["fc.bias"].shape[0]
            m.fc = nn.Linear(m.fc.in_features, outdim)
        m.load_state_dict(state)
    else:
        enum = getattr(ResNet18_Weights, MODEL_WEIGHTS)
        m = models.resnet18(weights=enum)
    return m.to(device).eval()

model = build_model()

# 3) Build idx→label map
METHODS = ["random","saliency","guided_backprop","smoothgrad",
           "gradcam","guided_gradcam","inputxgradient",
           "integrated_gradients","gradientshap"]

if WEIGHTS_PATH and LABEL_CFG["method"]=="folder":
    first = os.path.join(INPUT_FOLDER, METHODS[0])
    cats  = sorted(d for d in os.listdir(first)
                   if os.path.isdir(os.path.join(first, d)))
    idx_to_label = [c.lower() for c in cats]
else:
    import urllib.request
    url = ("https://raw.githubusercontent.com/pytorch/hub/"
           "master/imagenet_classes.txt")
    idx_to_label = [line.decode().strip().lower()
                    for line in urllib.request.urlopen(url)]

# 4) Transform + helpers
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),   # direct resize to 224×224
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
VALID_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def get_truth(path):
    if LABEL_CFG["method"] == "folder":
        # ascend two levels: .../<method>/<category>/qXX/file.png
        return os.path.basename(os.path.dirname(os.path.dirname(path))).lower()
    # filename mode
    fn    = os.path.splitext(os.path.basename(path))[0]
    parts = [tok for tok in fn.split(LABEL_CFG.get("delimiter","_")) if tok]
    return " ".join(parts).lower()

def load_img(path):
    return transform(Image.open(path).convert("RGB"))

# 5) Evaluate
results1 = {m: [] for m in METHODS}
results5 = {m: [] for m in METHODS}

for m in METHODS:
    print(f"\n=== Method: {m} ===")
    for pct in PCTS:
        paths = []
        base  = os.path.join(INPUT_FOLDER, m)
        if LABEL_CFG["method"]=="folder":
            for cat in os.listdir(base):
                qdir = os.path.join(base, cat, f"{pct:02d}")
                if not os.path.isdir(qdir): continue
                for f in os.listdir(qdir):
                    if os.path.splitext(f)[1].lower() in VALID_EXTS:
                        paths.append(os.path.join(qdir, f))
        else:
            qdir = os.path.join(base, f"q{pct:02d}")
            if os.path.isdir(qdir):
                for f in os.listdir(qdir):
                    if os.path.splitext(f)[1].lower() in VALID_EXTS:
                        paths.append(os.path.join(qdir, f))

        total = len(paths)
        if total == 0:
            print(f" pct={pct:02d}: no images")
            results1[m].append(None)
            results5[m].append(None)
            continue

        imgs   = list(ThreadPoolExecutor().map(load_img, paths))
        truths = [get_truth(p) for p in paths]

        # warm-up
        bs = min(128, total)
        _  = model(torch.stack(imgs[:bs]).to(device))

        t1c = t5c = 0
        for i in range(0, total, bs):
            batch = torch.stack(imgs[i:i+bs])\
                       .pin_memory().to(device, non_blocking=True)
            with torch.no_grad():
                out  = model(batch)
                idx1 = out.topk(1,dim=1)[1].cpu().squeeze(1).tolist()
                idx5 = out.topk(5,dim=1)[1].cpu().tolist()
            for j in range(len(idx1)):
                p1 = idx_to_label[idx1[j]]
                p5 = [idx_to_label[k] for k in idx5[j]]
                if truths[i+j] in p1:     t1c += 1
                if any(truths[i+j] in x for x in p5): t5c += 1

        acc1 = t1c/total
        acc5 = t5c/total
        print(f" pct={pct:02d}: Top-1={acc1:.3f}, Top-5={acc5:.3f} over {total}")
        results1[m].append(acc1)
        results5[m].append(acc5)

# 6) Write CSV + plots
with open(RESULTS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([f"XAI Eval ({MODEL_NAME})"])
    w.writerow([])
    w.writerow(["Top-1 Accuracy"])
    w.writerow(["pct"] + PCTS)
    for m in METHODS:
        w.writerow([m] + [f"{v:.3f}" if v is not None else "" 
                          for v in results1[m]])
    w.writerow([])
    w.writerow(["Top-5 Accuracy"])
    w.writerow(["pct"] + PCTS)
    for m in METHODS:
        w.writerow([m] + [f"{v:.3f}" if v is not None else "" 
                          for v in results5[m]])
print(f"Results → {RESULTS_CSV}")

def plot(data, title, fname):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    for m in METHODS:
        y = data[m]
        if any(v is None for v in y): continue
        plt.plot(PCTS, y, marker='o', label=m)
    plt.title(title); plt.xlabel("% pixels removed"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Plot → {fname}")

plot(results1, f"Top-1 ({MODEL_NAME})", f"{MODEL_NAME}_top1.png")
plot(results5, f"Top-5 ({MODEL_NAME})", f"{MODEL_NAME}_top5.png")
print("Done.")
