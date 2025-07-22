import os
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
import time

# ==================== CONFIGURATION PARAMETERS ====================
MASK_COLOR = (0.0, 0.0, 0.0)  # Black mask RGB (0-1)
CNN_WEIGHTS = ResNet18_Weights.DEFAULT
CNN_MODEL_FUNC = models.resnet18
GRADCAM_TARGET_LAYER = 'layer4'
REDUCE_SAMPLES = False
SKIP_SLOW_METHODS = False
BATCH_THRESHOLDS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Model: {CNN_MODEL_FUNC.__name__} with weights {CNN_WEIGHTS}")
print(f"Mask color: {MASK_COLOR}")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
MASK_COLOR_TENSOR = torch.tensor(MASK_COLOR, device=device).view(3, 1, 1)


def denormalize(t):
    return t * STD + MEAN


def save_tensor_img(tensor, path):
    if os.path.exists(path):
        print(f"Skipping existing file: {path}")
        return
    img = denormalize(tensor.squeeze(0))
    img = torch.clamp(img, 0, 1)
    img_np = img.cpu().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
    img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save(path, quality=95)


def mask_image_with_map(img_tensor_norm, mask, threshold):
    img_tensor = denormalize(img_tensor_norm.squeeze(0))

    mask = mask.squeeze(0).squeeze(0)
    mask_min = mask.min()
    mask_range = mask.max() - mask_min
    mask_norm = (mask - mask_min) / (mask_range + 1e-8)

    cutoff = torch.quantile(mask_norm.flatten(), threshold)
    binary_mask = (mask_norm > cutoff).to(dtype=img_tensor.dtype, device=img_tensor.device)

    total_pixels = binary_mask.numel()
    kept_pixels = binary_mask.sum().item()
    pixels_removed_pct = (1 - kept_pixels / total_pixels) * 100

    binary_mask_3c = binary_mask.unsqueeze(0).expand(3, -1, -1)
    masked_img = img_tensor * binary_mask_3c + MASK_COLOR_TENSOR * (1.0 - binary_mask_3c)
    masked_img_norm = (masked_img - MEAN) / STD

    return masked_img_norm.unsqueeze(0), pixels_removed_pct


def random_mask_with_threshold(img_tensor_norm, threshold):
    """
    Create a random mask using the same threshold logic as other methods.
    Generate a random importance map and apply the threshold quantile.
    """
    img_tensor = denormalize(img_tensor_norm.squeeze(0))
    _, H, W = img_tensor.shape

    # Create a random "importance" map (uniform random values)
    random_importance = torch.rand(H, W, device=img_tensor.device)

    # Apply the same threshold logic as other methods
    cutoff = torch.quantile(random_importance.flatten(), threshold)
    binary_mask = (random_importance > cutoff).to(dtype=img_tensor.dtype, device=img_tensor.device)

    total_pixels = binary_mask.numel()
    kept_pixels = binary_mask.sum().item()
    pixels_removed_pct = (1 - kept_pixels / total_pixels) * 100

    binary_mask_3c = binary_mask.unsqueeze(0).expand(3, -1, -1)
    masked_img = img_tensor * binary_mask_3c + MASK_COLOR_TENSOR * (1.0 - binary_mask_3c)
    masked_img_norm = (masked_img - MEAN) / STD

    return masked_img_norm.unsqueeze(0), pixels_removed_pct


class GuidedReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor):
        ctx.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_tensor < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input


class GuidedReLU(nn.Module):
    def forward(self, input_tensor):
        return GuidedReLUFunction.apply(input_tensor)


def guided_backprop_mask(model, img_tensor, class_idx):
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    with autocast():
        output = model(img_tensor)
    model.zero_grad()
    output[0, class_idx].backward()
    grad = img_tensor.grad.abs().sum(dim=1, keepdim=True)
    return grad


def smoothgrad_mask(model, img_tensor, class_idx, n_samples=None, noise_level=0.1):
    n_samples = 8 if REDUCE_SAMPLES else 15 if n_samples is None else n_samples
    avg_grad = torch.zeros_like(img_tensor)
    for _ in range(n_samples):
        noise = noise_level * torch.randn_like(img_tensor)
        noisy_img = img_tensor + noise
        noisy_img.requires_grad_()
        with autocast():
            out = model(noisy_img)
        model.zero_grad()
        out[0, class_idx].backward()
        avg_grad += noisy_img.grad.abs()
    avg_grad /= n_samples
    return avg_grad.sum(dim=1, keepdim=True)


def gradcam_mask(model, img_tensor, class_idx, cam_extractor):
    with autocast():
        output = model(img_tensor)
        cams = cam_extractor(class_idx, output)
    cam = cams[0] if isinstance(cams, list) else cams
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim() == 3:
        cam = cam.unsqueeze(0)
    return F.interpolate(cam.float(), size=(224, 224), mode='bilinear', align_corners=False)


def guided_gradcam_mask(model, img_tensor, class_idx, guided_cam_extractor):
    with autocast():
        output = model(img_tensor)
        cams = guided_cam_extractor(class_idx, output)
    cam = cams[0] if isinstance(cams, list) else cams
    if cam.dim() == 2:
        cam = cam.unsqueeze(0).unsqueeze(0)
    elif cam.dim() == 3:
        cam = cam.unsqueeze(0)
    return F.interpolate(cam.float(), size=(224, 224), mode='bilinear', align_corners=False)


def saliency_mask(model, img_tensor, class_idx, saliency_method):
    with autocast():
        attr = saliency_method.attribute(img_tensor, target=class_idx)
    return attr.abs().sum(dim=1, keepdim=True)


def inputxgradient_mask(model, img_tensor, class_idx, ixg_method):
    with autocast():
        attr = ixg_method.attribute(img_tensor, target=class_idx)
    return attr.abs().sum(dim=1, keepdim=True)


def integrated_gradients_mask(model, img_tensor, class_idx, ig_method, baseline):
    n_steps = 50 if REDUCE_SAMPLES else 100
    with autocast():
        attr = ig_method.attribute(img_tensor, baselines=baseline, target=class_idx, n_steps=n_steps)
    return attr.abs().sum(dim=1, keepdim=True)


def gradientshap_mask(model, img_tensor, class_idx, gs_method, baseline):
    n_samples = 15 if REDUCE_SAMPLES else 30
    with autocast():
        attr = gs_method.attribute(img_tensor, baselines=baseline, target=class_idx, n_samples=n_samples, stdevs=0.09)
    return attr.abs().sum(dim=1, keepdim=True)


def mask_multiple_thresholds(img_tensor_norm, mask, thresholds):
    img_tensor = denormalize(img_tensor_norm.squeeze(0))
    mask = mask.squeeze(0).squeeze(0)
    mask_min = mask.min()
    mask_range = mask.max() - mask_min
    mask_norm = (mask - mask_min) / (mask_range + 1e-8)

    results = {}
    # fix dtype of quantiles tensor to match mask_norm dtype:
    cutoffs = torch.quantile(mask_norm.flatten(), torch.tensor(thresholds, device=mask.device, dtype=mask_norm.dtype))

    for i, threshold in enumerate(thresholds):
        cutoff = cutoffs[i]
        binary_mask = (mask_norm > cutoff).to(dtype=img_tensor.dtype, device=img_tensor.device)

        total_pixels = binary_mask.numel()
        kept_pixels = binary_mask.sum().item()
        pixels_removed_pct = (1 - kept_pixels / total_pixels) * 100

        binary_mask_3c = binary_mask.unsqueeze(0).expand(3, -1, -1)
        masked_img = img_tensor * binary_mask_3c + MASK_COLOR_TENSOR * (1.0 - binary_mask_3c)
        masked_img_norm = (masked_img - MEAN) / STD

        results[threshold] = (masked_img_norm.unsqueeze(0), pixels_removed_pct)

    return results


def random_mask_multiple_thresholds(img_tensor_norm, thresholds):
    """
    Apply random masking for multiple thresholds efficiently.
    """
    img_tensor = denormalize(img_tensor_norm.squeeze(0))
    _, H, W = img_tensor.shape

    # Create a single random "importance" map for all thresholds
    random_importance = torch.rand(H, W, device=img_tensor.device)

    results = {}
    cutoffs = torch.quantile(random_importance.flatten(),
                             torch.tensor(thresholds, device=img_tensor.device, dtype=random_importance.dtype))

    for i, threshold in enumerate(thresholds):
        cutoff = cutoffs[i]
        binary_mask = (random_importance > cutoff).to(dtype=img_tensor.dtype, device=img_tensor.device)

        total_pixels = binary_mask.numel()
        kept_pixels = binary_mask.sum().item()
        pixels_removed_pct = (1 - kept_pixels / total_pixels) * 100

        binary_mask_3c = binary_mask.unsqueeze(0).expand(3, -1, -1)
        masked_img = img_tensor * binary_mask_3c + MASK_COLOR_TENSOR * (1.0 - binary_mask_3c)
        masked_img_norm = (masked_img - MEAN) / STD

        results[threshold] = (masked_img_norm.unsqueeze(0), pixels_removed_pct)

    return results


def process_all_methods(image_path, output_base_folder, thresholds):
    print(f"Processing image: {image_path}")
    start_time = time.time()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    model = CNN_MODEL_FUNC(weights=CNN_WEIGHTS).to(device).eval()
    with torch.no_grad(), autocast():
        pred = model(img_tensor)
    class_idx = pred.argmax(dim=1).item()

    guided_bp_model = CNN_MODEL_FUNC(weights=CNN_WEIGHTS).to(device).eval()

    def replace_relu_with_guided(module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, GuidedReLU())
            else:
                replace_relu_with_guided(child)

    replace_relu_with_guided(guided_bp_model)

    cam_extractor = GradCAM(model, target_layer=GRADCAM_TARGET_LAYER)
    guided_cam_extractor = SmoothGradCAMpp(model, target_layer=GRADCAM_TARGET_LAYER)
    saliency_method = Saliency(model)
    ixg_method = InputXGradient(model)
    gs_method = GradientShap(model)
    ig_method = IntegratedGradients(model)
    baseline = torch.zeros_like(img_tensor)

    methods = {
        "guided_backprop": lambda: guided_backprop_mask(guided_bp_model, img_tensor, class_idx),
        "smoothgrad": lambda: smoothgrad_mask(model, img_tensor, class_idx),
        "gradcam": lambda: gradcam_mask(model, img_tensor, class_idx, cam_extractor),
        "guided_gradcam": lambda: guided_gradcam_mask(model, img_tensor, class_idx, guided_cam_extractor),
        "saliency": lambda: saliency_mask(model, img_tensor, class_idx, saliency_method),
        "inputxgradient": lambda: inputxgradient_mask(model, img_tensor, class_idx, ixg_method),
        "integrated_gradients": lambda: integrated_gradients_mask(model, img_tensor, class_idx, ig_method, baseline),
        "gradientshap": lambda: gradientshap_mask(model, img_tensor, class_idx, gs_method,
                                                 baseline) if not SKIP_SLOW_METHODS else None,
        "random": lambda: torch.rand(1, 1, 224, 224, device=device),
        # Return a random "mask" that can be processed normally
    }

    results_log = []

    for method_name, compute_mask in methods.items():
        if compute_mask is None:
            print(f"Skipping {method_name} (disabled for speed)")
            continue

        method_start = time.time()

        if BATCH_THRESHOLDS:
            if method_name == "random":
                # Use the new random masking function
                threshold_results = random_mask_multiple_thresholds(img_tensor, thresholds)
            else:
                mask = compute_mask()
                threshold_results = mask_multiple_thresholds(img_tensor, mask, thresholds)

            for threshold, (masked_tensor, pixels_removed_pct) in threshold_results.items():
                out_dir = os.path.join(output_base_folder, method_name, f"{method_name}{int(threshold * 100):02d}")
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f"{image_name}_{method_name}{int(threshold * 100):02d}.jpg")

                if os.path.exists(save_path):
                    print(f"Already exists, skipping: {save_path}")
                    continue

                save_tensor_img(masked_tensor, save_path)
                result_info = f"Saved {save_path} | Pixels removed: {pixels_removed_pct:.1f}%"
                print(result_info)
                results_log.append(result_info)
        else:
            # Non-batch processing (fallback)
            for threshold in thresholds:
                out_dir = os.path.join(output_base_folder, method_name, f"{method_name}{int(threshold * 100):02d}")
                os.makedirs(out_dir, exist_ok=True)

                save_path = os.path.join(out_dir, f"{image_name}_{method_name}{int(threshold * 100):02d}.jpg")
                if os.path.exists(save_path):
                    print(f"Already exists, skipping: {save_path}")
                    continue

                if method_name == "random":
                    masked_tensor, pixels_removed_pct = random_mask_with_threshold(img_tensor, threshold)
                else:
                    mask = compute_mask()
                    masked_tensor, pixels_removed_pct = mask_image_with_map(img_tensor, mask, threshold)

                save_tensor_img(masked_tensor, save_path)
                result_info = f"Saved {save_path} | Pixels removed: {pixels_removed_pct:.1f}%"
                print(result_info)
                results_log.append(result_info)

        method_time = time.time() - method_start
        print(f"Method {method_name} completed in {method_time:.2f}s")

    total_time = time.time() - start_time
    print(f"Total processing time for {image_name}: {total_time:.2f}s")

    return results_log


def worker_process(img_path, output_folder, thresholds):
    try:
        results = process_all_methods(img_path, output_folder, thresholds)
        return f"Processed {img_path}\n" + "\n".join(results)
    except Exception as e:
        return f"Error processing {img_path}: {e}"


if __name__ == "__main__":
    input_folder = r"G:\My Drive\imagenet"
    output_folder = r"C:\imagenet_results"
    thresholds = [round(i * 0.05, 2) for i in range(1, 20)]

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)
                   if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    max_workers = min(4, os.cpu_count())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_process, img_path, output_folder, thresholds) for img_path in image_files]
        for future in as_completed(futures):
            print(future.result())