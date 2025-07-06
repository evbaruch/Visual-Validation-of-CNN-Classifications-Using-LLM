import os
import re
from PIL import Image

def is_image_okay(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Check if image can be opened and is not corrupted
        return True
    except Exception:
        return False

def remove_gradcam_suffix(root_directory):
    pattern = re.compile(r'(.*)_random\d+\.(jpg|png|jpeg)$', re.IGNORECASE)
    corrupted_file_count = 0
    renamed_file_count = 0
    skipped_file_count = 0
    ok_file_count = 0
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)
            if not is_image_okay(old_path):
                print(f"❌ {old_path} (corrupted or unreadable)")
                corrupted_file_count += 1
                continue
            match = pattern.match(filename)
            if match:
                new_filename = f"{match.group(1)}.{match.group(2)}"
                new_path = os.path.join(dirpath, new_filename)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)
                    print(f"➡️ {old_path} -> {new_path}")
                    renamed_file_count += 1
                else:
                    print(f"⚠️ Skipped (target exists): {new_path}")
                    skipped_file_count += 1
            else:
                print(f"✅ {old_path} (no rename needed)")
                ok_file_count += 1
    print(f"Total files corrupted ❌: {corrupted_file_count}")
    print(f"Total files renamed ➡️: {renamed_file_count}")
    print(f"Total files skipped ⚠️: {skipped_file_count}")
    print(f"Total files okay ✅: {ok_file_count}")

# Example usage:
remove_gradcam_suffix("data\\midAvi_grey\\random")