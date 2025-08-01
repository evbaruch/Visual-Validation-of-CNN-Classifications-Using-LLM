import os
import csv
import glob
from collections import defaultdict
from LmmApi.LLMInterface import LLMInterface
from LmmApi.llama32Vision11b import llama32Vision11b
from pydantic import BaseModel
from tqdm import tqdm


class CervicalCellPrediction(BaseModel):
    category: str


def get_few_shot_prompt_and_paths(few_shot_folder: str):
    prompt_lines = []
    image_paths = []

    for img_path in sorted(glob.glob(os.path.join(few_shot_folder, "*.jpeg"))):
        filename = os.path.basename(img_path)
        label = filename.split("_")[0].capitalize()
        prompt_lines.append(f"Image {filename} shows a {label} cell.")
        image_paths.append(img_path)

    prompt = "\n".join(prompt_lines)
    return prompt, image_paths


def collect_test_images(base_folder: str):
    grouped = defaultdict(lambda: defaultdict(list))
    for root, _, files in os.walk(base_folder):
        for file in files:
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = os.path.normpath(root).split(os.sep)
            # Expect: .../<base_folder>/<xai_method>/<percentage>/<label>/
            if len(parts) < 4:
                continue
            xai_method = parts[-4]
            percentage = parts[-3]
            label = parts[-2]
            full_path = os.path.join(root, file)
            grouped[xai_method][percentage].append({
                "path": full_path,
                "label": label
            })
    return grouped


import shutil
import tempfile

def classify_image(llm_context, few_shot_prompt, few_shot_paths, test_image_path, use_few_shot=True):
    prompt = (
        "You are a medical image analysis expert. Your task is to classify a cervical cell image "
        "into one of the following categories:\n\n"
        "- Dyskeratotic: Abnormal keratinization, hyperchromatic nuclei.\n"
        "- Koilocytotic: Perinuclear halo, nuclear enlargement, irregularity.\n"
        "- Metaplastic: Immature squamous cells, dense cytoplasm.\n"
        "- Parabasal: Small round cells with large nuclei, usually in clusters.\n"
        "- Superficial-Intermediate: Flattened cells with small nuclei, abundant cytoplasm.\n\n"
        "Few-shot examples:\n" + few_shot_prompt + "\n\nNow classify the next image."
    )

    llm_context.set_background(prompt)
    llm_context.set_jsonDescription(CervicalCellPrediction)

    if use_few_shot:
        # Few-shot: copy few-shot images + test image to temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            for img_path in few_shot_paths:
                shutil.copy(img_path, temp_dir)
            shutil.copy(test_image_path, temp_dir)
            llm_context.anchored_outputs_classification(
                root_directory=temp_dir,
                save_path="temp_llm_result"
            )
    else:
        # Single image: copy only test image to temp dir
        with tempfile.TemporaryDirectory() as temp_dir:
            shutil.copy(test_image_path, temp_dir)
            llm_context.anchored_outputs_classification(
                root_directory=temp_dir,
                save_path="temp_llm_result"
            )

    # Read prediction
    result_path = os.path.join("temp_llm_result", "response.json")
    if not os.path.exists(result_path):
        return None

    import json
    with open(result_path, "r") as f:
        try:
            prediction = json.load(f)
            return prediction.get("category", "").strip().capitalize()
        except:
            return None

def build_accuracy_matrix(few_shot_folder, test_folder, output_csv_path):
    few_shot_prompt, few_shot_paths = get_few_shot_prompt_and_paths(few_shot_folder)
    grouped = collect_test_images(test_folder)
    llama = llama32Vision11b()
    llm_context = LLMInterface(llama)

    matrix = defaultdict(dict)

    for xai_method in grouped:
        for percentage in grouped[xai_method]:
            total = 0
            correct = 0
            print(f"\n🔎 Running: {xai_method} | {percentage} | {len(grouped[xai_method][percentage])} images")
            for image_data in tqdm(grouped[xai_method][percentage]):
                real_label = image_data["label"].capitalize()
                test_path = image_data["path"]

                predicted = classify_image(llm_context, few_shot_prompt, few_shot_paths, test_path)
                if predicted:
                    total += 1
                    if predicted == real_label:
                        correct += 1

            acc = (correct / total) if total else 0.0
            matrix[xai_method][percentage] = round(acc, 3)

    # Write to CSV
    all_percentages = sorted({p for v in matrix.values() for p in v})
    with open(output_csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + all_percentages)
        for xai_method in matrix:
            row = [xai_method] + [matrix[xai_method].get(p, "") for p in all_percentages]
            writer.writerow(row)

    print(f"\n✅ Accuracy matrix saved to: {output_csv_path}")


if __name__ == "__main__":
    build_accuracy_matrix(
        few_shot_folder="data\\few_shots\\",
        test_folder="data\\mid_CervicalCancer_evtr\\",
        output_csv_path="llm_accuracy_matrix.csv"
    )
