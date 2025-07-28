import os
import shutil
import random

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

# 2. Gather all candidate images per XAI method (for both masks)
xai_candidates = {}
for xai_method in os.listdir(xai_base):
    method_candidates = []
    for mask in mask_folders:
        mask_dir = os.path.join(xai_base, xai_method, mask)
        if not os.path.isdir(mask_dir):
            continue
        for root, _, files in os.walk(mask_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    method_candidates.append(os.path.join(root, file))
    if method_candidates:
        xai_candidates[xai_method] = method_candidates

# 3. Calculate how many to take from each method
num_methods = len(xai_candidates)
if num_methods == 0:
    raise RuntimeError("No XAI methods found with images.")
per_method = (orig_count * 2) // num_methods

# 4. Randomly select up to per_method images from each method
selected = []
for xai_method, candidates in xai_candidates.items():
    random.shuffle(candidates)
    selected.extend(candidates[:per_method])

# If due to rounding we have less than orig_count*2, fill up with leftovers
if len(selected) < orig_count * 2:
    leftovers = []
    for xai_method, candidates in xai_candidates.items():
        leftovers.extend(candidates[per_method:])
    random.shuffle(leftovers)
    selected.extend(leftovers[:(orig_count * 2 - len(selected))])

# 5. Copy selected images to output_dir, preserving XAI/mask/label structure
for path in selected:
    rel_path = os.path.relpath(path, xai_base)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(path, out_path)

# 6. Copy all original images as well
for orig_path in orig_images:
    rel_path = os.path.relpath(orig_path, src_dir)
    out_path = os.path.join(output_dir, "original", rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    shutil.copy2(orig_path, out_path)

print(f"Done! Copied {len(selected)} masked images (equal per XAI method) and all {orig_count} original images to {output_dir}")