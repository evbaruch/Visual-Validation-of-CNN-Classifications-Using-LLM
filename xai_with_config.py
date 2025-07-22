# xai_with_config.py
import os
import sys
import json
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from captum.attr import Saliency, InputXGradient, GradientShap, IntegratedGradients
from torchcam.methods import GradCAM, SmoothGradCAMpp
from torch.cuda.amp import autocast
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --------------------
# 1) Load config
# --------------------
if len(sys.argv) != 2:
    print("Usage: python xai_with_config.py <config.json>")
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    cfg = json.load(f)

# Paths & model source
INPUT_FOLDER       = cfg["input_folder"]
OUTPUT_FOLDER      = cfg["mid_folder"]
MODEL_NAME         = cfg["model_name"]            # e.g. "resnet18"
MODEL_WEIGHTS      = cfg.get("model_weights")     # e.g. "DEFAULT" or null
WEIGHTS_PATH       = cfg.get("weights_path")      # e.g. "/mnt/data/cancer_resnet18.pth"
NUM_CLASSES        = cfg["num_classes"]

# XAI / masking parameters
MASK_COLOR         = tuple(cfg["mask_color"])
GRADCAM_TARGET_LAYER = cfg["gradcam_target_layer"]
REDUCE_SAMPLES     = cfg["reduce_samples"]
SKIP_SLOW_METHODS  = cfg["skip_slow_methods"]
BATCH_THRESHOLDS   = cfg["batch_thresholds"]
THRESHOLDS         = cfg["thresholds"]
IMG_SIZE           = cfg["img_size"]

# Normalization
NORMALIZE_MEAN     = cfg["normalize_mean"]
NORMALIZE_STD      = cfg["normalize_std"]

# How to get the true category for each image
LABEL_CFG          = cfg["label_extraction"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------
# 2) Transforms & constants
# --------------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

MEAN_T = torch.tensor(NORMALIZE_MEAN, device=device).view(3,1,1)
STD_T  = torch.tensor(NORMALIZE_STD,  device=device).view(3,1,1)
MASK_T = torch.tensor(MASK_COLOR,      device=device).view(3,1,1)

def denormalize(x: torch.Tensor) -> torch.Tensor:
    return x * STD_T + MEAN_T

def save_tensor_img(tensor: torch.Tensor, path: str):
    img = denormalize(tensor.squeeze(0)).clamp(0,1)
    arr = (img.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, quality=95)

# --------------------
# 3) All masking / attribution functions
# --------------------
def mask_image_with_map(img_tensor_norm, mask, threshold):
    img = denormalize(img_tensor_norm.squeeze(0))
    m = mask.squeeze(0).squeeze(0)
    mmin, mmax = m.min(), m.max()
    m = (m - mmin) / (mmax - mmin + 1e-8)
    cutoff = torch.quantile(m.flatten(), threshold)
    binary = (m > cutoff).to(dtype=img.dtype, device=img.device)
    binary3 = binary.unsqueeze(0).expand_as(img)
    masked = img * binary3 + MASK_T * (1 - binary3)
    masked_norm = (masked - MEAN_T) / STD_T
    removed = (1 - binary.sum().item() / binary.numel()) * 100
    return masked_norm.unsqueeze(0), removed

def random_mask_with_threshold(img_tensor_norm, threshold):
    img = denormalize(img_tensor_norm.squeeze(0))
    _, H, W = img.shape
    rnd = torch.rand(H, W, device=img.device)
    cutoff = torch.quantile(rnd.flatten(), threshold)
    binary = (rnd > cutoff).to(dtype=img.dtype, device=img.device)
    binary3 = binary.unsqueeze(0).expand_as(img)
    masked = img * binary3 + MASK_T * (1 - binary3)
    masked_norm = (masked - MEAN_T) / STD_T
    removed = (1 - binary.sum().item() / binary.numel()) * 100
    return masked_norm.unsqueeze(0), removed

class GuidedReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_in = grad_output.clone()
        grad_in[inp < 0]   = 0
        grad_in[grad_output < 0] = 0
        return grad_in

class GuidedReLU(nn.Module):
    def forward(self, x): return GuidedReLUFunction.apply(x)

def guided_backprop_mask(model, x, cls):
    x = x.clone().detach().requires_grad_(True)
    with autocast():
        out = model(x)
    model.zero_grad()
    out[0, cls].backward()
    return x.grad.abs().sum(dim=1, keepdim=True)

def smoothgrad_mask(model, x, cls, n_samples=None, noise_level=0.1):
    n = 8 if REDUCE_SAMPLES else (n_samples or 15)
    avg_grad = torch.zeros_like(x)
    for _ in range(n):
        noise = noise_level * torch.randn_like(x)
        nx = (x + noise).requires_grad_()
        with autocast():
            out = model(nx)
        model.zero_grad()
        out[0, cls].backward()
        avg_grad += nx.grad.abs()
    avg_grad /= n
    return avg_grad.sum(dim=1, keepdim=True)

def gradcam_mask(model, x, cls, cam_ext):
    with autocast():
        out = model(x)
        cams = cam_ext(cls, out)
    cam = cams[0] if isinstance(cams, list) else cams
    if cam.dim()==2: cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim()==3: cam = cam.unsqueeze(0)
    return F.interpolate(cam.float(), size=(IMG_SIZE,IMG_SIZE),
                         mode='bilinear', align_corners=False)

def guided_gradcam_mask(model, x, cls, guided_ext):
    with autocast():
        out = model(x)
        cams = guided_ext(cls, out)
    cam = cams[0] if isinstance(cams, list) else cams
    if cam.dim()==2: cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim()==3: cam = cam.unsqueeze(0)
    return F.interpolate(cam.float(), size=(IMG_SIZE,IMG_SIZE),
                         mode='bilinear', align_corners=False)

def saliency_mask(model, x, cls, sal):
    with autocast():
        attr = sal.attribute(x, target=cls)
    return attr.abs().sum(dim=1, keepdim=True)

def inputxgradient_mask(model, x, cls, ixg):
    with autocast():
        attr = ixg.attribute(x, target=cls)
    return attr.abs().sum(dim=1, keepdim=True)

def integrated_gradients_mask(model, x, cls, ig, baseline):
    steps = 50 if REDUCE_SAMPLES else 100
    with autocast():
        attr = ig.attribute(x, baselines=baseline, target=cls, n_steps=steps)
    return attr.abs().sum(dim=1, keepdim=True)

def gradientshap_mask(model, x, cls, gs, baseline):
    samples = 15 if REDUCE_SAMPLES else 30
    with autocast():
        attr = gs.attribute(x, baselines=baseline,
                            target=cls, n_samples=samples, stdevs=0.09)
    return attr.abs().sum(dim=1, keepdim=True)

def mask_multiple_thresholds(x_norm, mask, thresholds):
    img = denormalize(x_norm.squeeze(0))
    m = mask.squeeze(0).squeeze(0)
    mmin, mmax = m.min(), m.max()
    m = (m - mmin) / (mmax - mmin + 1e-8)
    cutoffs = torch.quantile(m.flatten(),
                             torch.tensor(thresholds, device=m.device, dtype=m.dtype))
    out = {}
    for i, t in enumerate(thresholds):
        cutoff = cutoffs[i]
        binary = (m > cutoff).to(dtype=img.dtype, device=img.device)
        binary3 = binary.unsqueeze(0).expand_as(img)
        masked = img * binary3 + MASK_T * (1-binary3)
        masked_norm = (masked - MEAN_T) / STD_T
        removed = (1 - binary.sum().item()/binary.numel())*100
        out[t] = (masked_norm.unsqueeze(0), removed)
    return out

def random_mask_multiple_thresholds(x_norm, thresholds):
    img = denormalize(x_norm.squeeze(0))
    _, H, W = img.shape
    rnd = torch.rand(H, W, device=img.device)
    cutoffs = torch.quantile(rnd.flatten(),
                             torch.tensor(thresholds, device=img.device, dtype=rnd.dtype))
    out = {}
    for i, t in enumerate(thresholds):
        cutoff = cutoffs[i]
        binary = (rnd > cutoff).to(dtype=img.dtype, device=img.device)
        binary3 = binary.unsqueeze(0).expand_as(img)
        masked = img * binary3 + MASK_T * (1-binary3)
        masked_norm = (masked - MEAN_T) / STD_T
        removed = (1 - binary.sum().item()/binary.numel())*100
        out[t] = (masked_norm.unsqueeze(0), removed)
    return out

# --------------------
# 4) Label extraction
# --------------------
def get_category_from_path(path):
    m = LABEL_CFG["method"]
    if m == "folder":
        # parent‐folder name
        return os.path.basename(os.path.dirname(path))
    elif m == "filename":
        # filename without extension
        return os.path.basename(path)
    else:
        raise ValueError(f"Unknown label_extraction method: {m}")


# --------------------
# 5) Model builders
# --------------------
# def build_base_model():
#     if MODEL_WEIGHTS:
#         enum = getattr(ResNet18_Weights, MODEL_WEIGHTS)
#         m = getattr(models, MODEL_NAME)(weights=enum)
#     else:
#         m = getattr(models, MODEL_NAME)(pretrained=False)
#         m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
#         state = torch.load(WEIGHTS_PATH, map_location=device)
#         m.load_state_dict(state)
#     return m.to(device).eval()

def build_base_model():
    """
    Builds a ResNet‐18. If WEIGHTS_PATH is set it will:
      1) instantiate without pretrained weights,
      2) re-create the same Sequential head you used in training,
      3) load your checkpoint (fc.0, fc.3 keys will now match).
    Otherwise it falls back to a torchvision‐pretrained model.
    """
    if WEIGHTS_PATH:
        # 1) bare ResNet-18
        model = models.resnet18(pretrained=False)
        # 2) rebuild your trained head
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, NUM_CLASSES)
        )
        # 3) load your checkpoint
        state = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
    else:
        # ImageNet‐pretrained path
        weights_enum = getattr(ResNet18_Weights, MODEL_WEIGHTS)
        model = getattr(models, MODEL_NAME)(weights=weights_enum)
    return model.to(device).eval()

def build_guided_model():
    m = build_base_model()
    def _replace(module):
        for name, ch in module.named_children():
            if isinstance(ch, nn.ReLU):
                setattr(module, name, GuidedReLU())
            else:
                _replace(ch)
    _replace(m)
    return m

# --------------------
# 6) Core processing
# --------------------
def process_all_methods(img_path):
    cat = get_category_from_path(img_path)
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")
    x   = preprocess(img).unsqueeze(0).to(device)

    # models
    model        = build_base_model()
    guided_model = build_guided_model()

    # predict
    with torch.no_grad(), autocast():
        logits = model(x)
    class_idx = logits.argmax(dim=1).item()

    # XAI tools
    cam_ext        = GradCAM(model, target_layer=GRADCAM_TARGET_LAYER)
    guided_ext     = SmoothGradCAMpp(model, target_layer=GRADCAM_TARGET_LAYER)
    sal            = Saliency(model)
    ixg            = InputXGradient(model)
    ig             = IntegratedGradients(model)
    gs             = GradientShap(model)
    baseline       = torch.zeros_like(x)

    methods = {
      "guided_backprop":    lambda: guided_backprop_mask(guided_model, x, class_idx),
      "smoothgrad":         lambda: smoothgrad_mask(model, x, class_idx),
      "gradcam":            lambda: gradcam_mask(model, x, class_idx, cam_ext),
      "guided_gradcam":     lambda: guided_gradcam_mask(model, x, class_idx, guided_ext),
      "saliency":           lambda: saliency_mask(model, x, class_idx, sal),
      "inputxgradient":     lambda: inputxgradient_mask(model, x, class_idx, ixg),
      "integrated_gradients": lambda: integrated_gradients_mask(model, x, class_idx, ig, baseline),
      "gradientshap":       (lambda: gradientshap_mask(model, x, class_idx, gs, baseline))
                                 if not SKIP_SLOW_METHODS else None,
      "random":             lambda: torch.rand(1,1,IMG_SIZE,IMG_SIZE,device=device),
    }

    for xai_method, fn in methods.items():
        if fn is None:
            print(f"Skipping {xai_method}")
            continue

        if BATCH_THRESHOLDS and xai_method != "random":
            mask = fn()
            results = mask_multiple_thresholds(x, mask, THRESHOLDS)
        elif BATCH_THRESHOLDS and xai_method == "random":
            results = random_mask_multiple_thresholds(x, THRESHOLDS)
        else:
            # fallback single-threshold (not shown for brevity)
            continue

        for t, (xm, pct) in results.items():
            m = LABEL_CFG["method"]
            if m == "folder":
                # parent‐folder name
                out_dir = os.path.join(OUTPUT_FOLDER, xai_method, cat, f"{int(t*100):02d}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{img_name}.png")
                save_tensor_img(xm, out_path)
            elif m == "filename":
                
                out_dir = os.path.join(OUTPUT_FOLDER, xai_method, f"{int(t*100):02d}")
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{cat}.png")
                save_tensor_img(xm, out_path)

            # print(f"[{cat}][{name}@{int(t*100)}%] → {out_path} ({pct:.1f}% removed)")

# --------------------
# 7) Main loop
# --------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # recursively collect all images
    images = []
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for fn in files:
            if fn.lower().endswith((".jpg",".jpeg",".png", ".JPEG")):
                images.append(os.path.join(root, fn))

    print(f"Found {len(images)} images under {INPUT_FOLDER}")
    with ThreadPoolExecutor(max_workers=4) as exec:
        futures = [exec.submit(process_all_methods, p) for p in images]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Processing images"):
            try:
                fut.result()
            except Exception as e:
                print(f"ERROR processing image: {e}")
