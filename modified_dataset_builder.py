import os
import shutil
import random

# Source and random-masked directories
src_dir = r"data\source\CervicalCancer\JPEG\CROPPED"
rand_dir = r"data\mid_random\random"

# Always use these mask folders
mask_folders = [f"{i:02d}" for i in range(5, 15, 5)]  # ['05', '10', '15', '20', '25']

# Output: dictionary mapping original image path to list of 5 specific masked image paths
image_map = {}

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        rel_dir = os.path.relpath(root, src_dir)
        rel_path = os.path.join(rel_dir, file) if rel_dir != '.' else file

        # # Build the 5 specific random-masked paths for this image
        # chosen = []
        # for mask in mask_folders:
        #     masked_path = os.path.join(rand_dir, mask, rel_path)
        #     if os.path.exists(masked_path):
        #         chosen.append(masked_path)
        #     else:
        #         print(f"Warning: Masked version missing: {masked_path}")
        # if len(chosen) < 5:
        #     print(f"Warning: Less than 5 masked versions for {rel_path}")
        #     continue
        # image_map[os.path.join(root, file)] = chosen
        
        # Choose one random mask folder for this image
        mask = random.choice(mask_folders)
        masked_path = os.path.join(rand_dir, mask, rel_path)
        if os.path.exists(masked_path):
            image_map[os.path.join(root, file)] = [masked_path]
        else:
            print(f"Warning: Masked version missing: {masked_path}")

        

# save the modified dataset to a new folder near the original
output_dir = r"data\source\CervicalCancer\JPEG\CROPPED_modified_1-2"
os.makedirs(output_dir, exist_ok=True)

# Copy the selected images to the new folder, preserving structure and indicating mask
for orig_path, masked_list in image_map.items():
    rel_dir = os.path.relpath(os.path.dirname(orig_path), src_dir)
    base_name = os.path.basename(orig_path).split('.')[0]  # without extension
    for masked_path in masked_list:
        # Extract mask folder (e.g., "05") for naming
        mask_folder = os.path.basename(os.path.dirname(os.path.dirname(masked_path)))
        # Build output subdirectory
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        # Name: original name with mask prefix, e.g., "001_01_05.jpeg"
        out_name = f"{base_name}_{mask_folder}.jpeg"
        out_path = os.path.join(out_subdir, out_name)
        shutil.copy2(masked_path, out_path)
        
# Also copy all original images to the modified folder, preserving structure and extension
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        rel_dir = os.path.relpath(root, src_dir)
        out_subdir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_subdir, exist_ok=True)
        src_path = os.path.join(root, file)
        out_path = os.path.join(out_subdir, file)
        shutil.copy2(src_path, out_path)

print("Done! Masked images (05, 10, 15, 20, 25) per original have been copied to", output_dir)