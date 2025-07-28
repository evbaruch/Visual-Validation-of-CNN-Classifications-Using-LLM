import os
import shutil
import random
from glob import glob

src_dir = r"data\source\CervicalCancer\JPEG\CROPPED"
xai_base = r"data\mid_CervicalCancer_modified_dataset"
mask_folders = ["05", "10"]
output_dir = r"data\source\CervicalCancer\JPEG\CROPPED_modified_doubled"
os.makedirs(output_dir, exist_ok=True)

# 1. Count original images
orig_images = []
for root, _, files in os.walk(src_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            orig_images.append(os.path.join(root, file))
orig_count = len(orig_images)

# 2. Gather all candidate images from all XAI methods and mask folders
candidates = []
for xai_method in os.listdir(xai_base):
    for mask in mask_folders:
        mask_dir = os.path.join(xai_base, xai_method, mask)
        if not os.path.isdir(mask_dir):
            continue
        for root, _, files in os.walk(mask_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    candidates.append(os.path.join(root, file))

# 3. Randomly select up to orig_count * 2 images (no more)
random.shuffle(candidates)
selected = candidates[:orig_count * 2]

# 4. Copy selected images to output_dir, preserving XAI/mask/label structure
for path in selected:
    rel_path = os.path.relpath(path, xai_base)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(path, out_path)

# 5. Copy all original images as well
for orig_path in orig_images:
    rel_path = os.path.relpath(orig_path, src_dir)
    out_path = os.path.join(output_dir, "original", rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(orig_path, out_path)

print(f"Done! Copied {len(selected)} masked images and all {orig_count} original images to {output_dir}")