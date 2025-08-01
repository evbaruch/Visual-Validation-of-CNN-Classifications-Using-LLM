from LmmApi.LLMStrategy import LLMStrategy # this is the abstract class implementing the Stragegy pattern
from trash import global_data as gd # this file contains 
from pydantic import BaseModel # a base class for data validation and settings management using python type hints
import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sys
# Context Class
class LLMInterface:
    """
    this class is the Context class in the Strategy pattern 
    meaning it is the class that uses the Strategy interface to call the appropriate algorithm
    the class is responsible for setting the strategy and calling the algorithm
    the class is also responsible for processing the response from the LLM and returning the closest label
    the class is also responsible for saving the classification results to a CSV file 
    """
    
    def __init__(self, strategy: LLMStrategy): 
        """
        when the class is instantiated, a strategy is passed to it
        the strategies are the different classes that implement api calls to different LLMs
        """
        self.strategy = strategy 
        self.imagenet_categories = gd.load_imagenet_classes()


    def set_strategy(self, strategy: LLMStrategy):
        """Set a different strategy at runtime."""
        self.strategy = strategy

    def set_prompt(self, prompt: str):
        """Set the prompt for the LLM."""
        self.prompt = prompt

    def set_background(self, background: str):
        """Set the background for the LLM."""
        self.background = background

    def set_jsonDescription(self, jsonDescription: BaseModel):
        """Set the JSON description for the LLM."""
        self.jsonDescription = jsonDescription
    

    def structured_outputs_classification(self, root_directory: str, save_path: str, rewrite = False):
        """
        Classifies images in the specified root directory using a language model (LLM) strategy.
        The classification results are saved in CSV files, with one CSV file per method (folder name).

        Parameters:
        - root_directory (str): The root directory containing subdirectories with images to classify.
        - save_path (str): The directory where the CSV files with classification results will be saved.
        """
        os.makedirs(save_path, exist_ok=True)

        for dirpath, _, filenames in os.walk(root_directory): 
            image_files = [f for f in filenames if f.endswith('.png')] # f is the image_file name - replace 'f' with 'image_file'
            data = [] # this uses as the matrix of responses saved to the CSV file - rename to 'matrix_of_responses'
            max_labels = 0 # this is the maximum number of labels that used for what? - needs a better name and comment to explain the use
            max_llm = 0 # this is the maximum number of llm that used for what? - needs a better name and comment to explain the use

            method = os.path.basename(dirpath)
            save_path_csv = os.path.join(save_path, f"{method}.csv")
            if os.path.isfile(save_path_csv) and not rewrite:
                continue
        
            for file in tqdm(image_files, desc=f"Processing files in {dirpath}"):
                # Get the full path of the image file
                file_path = os.path.join(dirpath, file)


                # Generate a response from the LLM
                response = self.strategy.generate_response(self.prompt, file_path, self.jsonDescription)

                # Extract the model dump from the response
                response_list = list(response.model_dump().values()) # maybe rename to 'response_values'
                
                # Find the top 5 predicted labels
                labels = [] #predicted_labels
                for item in response_list:
                    _, closest_label = gd.find_closest_category(item, self.imagenet_categories) # TODO: understand what this function does
                    labels.append(closest_label)
                
                # Update the maximum number of labels
                max_labels = max(max_labels, len(labels)) # max_labels is the maximum number of labels returned from the llm? if so the proper name will be max_label_detected
                max_llm = max(max_llm, len(response_list)) # max_llm is the maximum number of values return from the llm? if so the proper name will be max_response_values

                # Extract image index, true label, and method (directory name)
                # recomended to make this a function at the file that contains the data processing functions
                # the function will take the file name and return the index, true label and method
                # it will be called 'file_name_interpreter' 
                # in the documentation it will be explained that the files name are structured in a way that the function can extract the index, true label and method
                # file_name example: 'resnet18/0_goldfish.png' - the function will return '0', 'goldfish', 'resnet18'
                # mid - is not a good name for the variable - rename to 'true_label_and_extention'
                index, mid = file.split('_')[:2]
                true_label = mid.split('.')[0]
                

                # Check if the true label is among the top 5 predicted labels
                correctly = true_label in labels # correctly is not the right name for this variable - rename to 'is_true_label_detected'

                # Append the results to the matrix of responses
                data.append([index, true_label, correctly] + labels + response_list)

            if len(data) == 0:
                continue

            label_columns = [f'Class_{i+1}' for i in range(max_labels)]
            llm_columns = [f'llm_{i+1}' for i in range(max_llm)] # llm_respondes_columns
            columns = ['Index', 'True_Label', 'Match'] + label_columns + llm_columns
            # Create a DataFrame from the array and add the column titles
            df = pd.DataFrame(data, columns=columns)

            # Save the DataFrame to a CSV file
            df.to_csv(save_path_csv, index=False)

    def anchored_outputs_classification(self, root_directory: str, save_path: str, rewrite = False):
            """
            Classifies images in the specified root directory using a language model (LLM) strategy.
            The classification results are saved in CSV files, with one CSV file per method (folder name).

            Parameters:
            - root_directory (str): The root directory containing subdirectories with images to classify.
            - save_path (str): The directory where the CSV files with classification results will be saved.
            """
            os.makedirs(save_path, exist_ok=True)

            for dirpath, _, filenames in os.walk(root_directory): 
                image_files = [f for f in filenames if f.endswith('.png')] 
                data = []
                max_labels = 0
                max_llm = 0

                method = os.path.basename(dirpath)
                save_path_csv = os.path.join(save_path, f"{method}.csv")
                if os.path.isfile(save_path_csv) and not rewrite:
                    continue
                
                for file in tqdm(image_files, desc=f"Processing files in {dirpath}"):
                    # Get the full path of the image file
                    image_path = os.path.join(dirpath, file)
                    image_name = os.path.basename(image_path)
                    image_name = image_name.split('_')[1].split('.')[0]  # Extracts 'tench'

                    # Generate a response from the LLM
                    response = self.strategy.generate_response(f"{self.prompt} Is this a {image_name}?", image_path, self.jsonDescription)

                    # Extract the model dump from the response
                    response_list = list(response.model_dump().values())

                    # Find the top 5 predicted labels
                    labels = []
                    for item in response_list:
                        _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
                        labels.append(closest_label)
                    
                    # Update the maximum number of labels
                    max_labels = max(max_labels, len(labels))
                    max_llm = max(max_llm, len(response_list))

                    # Extract image index, true label, and method (directory name)
                    index, mid = file.split('_')[:2]
                    true_label = mid.split('.')[0]
                    method = os.path.basename(dirpath)

                    # Check if the true label is among the top 5 predicted labels
                    correctly = true_label in labels

                    data.append([index, true_label, correctly] + labels + response_list)

                if len(data) == 0:
                    continue

                label_columns = [f'Class_{i+1}' for i in range(max_labels)]
                llm_columns = [f'llm_{i+1}' for i in range(max_llm)]
                columns = ['Index', 'True_Label', 'Match'] + label_columns + llm_columns
                # Create a DataFrame from the array and add the column titles
                df = pd.DataFrame(data, columns=columns)

                # Save the DataFrame to a CSV file
                df.to_csv(os.path.join(save_path, f"{method}.csv"), index=False)

    def boolean_outputs_classification(self, root_directory: str, save_path: str, rewrite = False):
        """
        Classifies images in the specified root directory using a language model (LLM) strategy.
        The classification results are saved in CSV files, with one CSV file per method (folder name).

        Parameters:
        - root_directory (str): The root directory containing subdirectories with images to classify.
        - save_path (str): The directory where the CSV files with classification results will be saved.
        """
        os.makedirs(save_path, exist_ok=True)

        for dirpath, _, filenames in os.walk(root_directory): 
            image_files = [f for f in filenames if f.endswith('.jpg')]  # Include common image formats
            data = []
            max_labels = 0
            max_llm = 0

            method = os.path.basename(dirpath)
            save_path_csv = os.path.join(save_path, f"{method}.csv")
            if os.path.isfile(save_path_csv) and not rewrite:
                continue
        

            for file in tqdm(image_files, desc=f"Processing files in {dirpath}"):
                # Get the full path of the image file
                file_path = os.path.join(dirpath, file)
                image_name = os.path.basename(file_path)
                image_name = image_name.split('_', 1)[1].split('.')[0]  # Extracts 'tench'
                image_name = image_name.replace('_', ' ')  # Replace underscores with spaces for better readability

                # Construct the prompt dynamically
                # prompt = (
                #     f"You are a medical image analysis expert. "
                #     f"Analyze the image and answer: Is it a {image_name}? "
                #     f"Use the following background knowledge: {self.background}"
                # )

                prompt = f"What do you see in the picture? Is it a {image_name} from the imagenet database?"

                # Generate a response from the LLM
                response = self.strategy.generate_response(
                    prompt=prompt,
                    background=self.background,
                    image=file_path,
                    jsonDescription=self.jsonDescription
                )
                # Extract the model dump from the response
                response = list(response.model_dump().values())

                # Find the top 5 predicted labels
                # labels = []
                # for item in response:
                #     _, closest_label = gd.find_closest_category(item, self.imagenet_categories)
                #     labels.append(closest_label)
                
                # Update the maximum number of labels
                max_labels = 0
                max_llm = 1

                # Extract image index, true label, and method (directory name)
                index, mid = file.split('_')[:2]
                true_label = mid.split('.')[0]
                

                # Check if the true label is among the top 5 predicted labels
                correctly = response[0]

                data.append([index, true_label, correctly] + response)

            if len(data) == 0:
                continue

            label_columns = [f'Class_{i+1}' for i in range(max_labels)]
            llm_columns = [f'llm_{i+1}' for i in range(max_llm)]
            columns = ['Index', 'True_Label', 'Match'] + label_columns + llm_columns
            # Create a DataFrame from the array and add the column titles
            df = pd.DataFrame(data, columns=columns)

            # Save the DataFrame to a CSV file
            df.to_csv(save_path_csv, index=False)
            
    # def boolean_outputs_classification_reverse(self, root_directory: str, save_path: str, rewrite = False):

    #     xai = "guided_gradcam"
    #     P = "05"
    #     columns = ['Index', 'True_Label', 'Match', 'llm_1']
        
    #     for image in os.listdir(f"{root_directory}/{xai}/{xai}{P}"):
    #         for P in ["05","10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60", "65", "70", "75", "80", "85", "90", "95"]:
    #             for xai in ["guided_backprop","guided_gradcam", "integrated_gradients","inputxgradient", "smoothgrad"]:
    #                 print(f"{root_directory}/{xai}/{xai}{P}/{image}")
    #                 file_path = f"{root_directory}/{xai}/{xai}_{P}/{image}"
    #                 basename = os.path.basename(file_path)
    #                 image_name = basename.split('_', 1)[1].split('.')[0]  # Extracts 'tench'
    #                 insex = basename.split('_')[0]  # Extracts the index from the file name
    #                 image_name = image_name.replace('_', ' ')  # Replace underscores with spaces for better readability
    #                 print(f"Image name: {image_name}")
                    
    #                 prompt = f"What do you see in the picture? Is it a {image_name} from the imagenet database?"

    #                 # Generate a response from the LLM
    #                 response = self.strategy.generate_response(
    #                     prompt=prompt,
    #                     background=self.background,
    #                     image=file_path,
    #                     jsonDescription=self.jsonDescription
    #                 )
    #                 # Extract the model dump from the response
    #                 response = list(response.model_dump().values())

    #                 correctly = response[0]
                    
                    
    #                 data = [insex, image_name, correctly] + response
                    
    #                     # add data to xai csv in a new line
                   
                   
                        
    def boolean_outputs_classification_reverse(self, root_directory: str, save_path: str, rewrite=False):
        """
        Stream-style processing with loop order: image -> P -> xai.
        Only path pattern: {root_directory}/{xai}/{xai}_{P}/{image}.
        Each image is saved immediately per (xai,P) CSV.
        """

        os.makedirs(save_path, exist_ok=True)
        
        xai = "guided_gradcam"
        P = "05"

        xai_methods = [
            "guided_backprop",
            "guided_gradcam",
            "integrated_gradients",
            "inputxgradient",
            "smoothgrad",
        ]
        P_values = ["05", "10", "15", "20", "25", "30", "35", "40", "45", "50",
                    "55", "60", "65", "70", "75", "80", "85", "90", "95"]


        def append_row(xai: str, P: str, row: list):
            combo = f"{xai}_{P}"
            csv_path = os.path.join(f"{save_path}/{xai}", f"{combo}.csv")
            os.makedirs(f"{save_path}/{xai}", exist_ok=True)
            llm_count = max(1, len(row) - 3)
            llm_columns = [f'llm_{i+1}' for i in range(llm_count)]
            columns = ['Index', 'True_Label', 'Match'] + llm_columns

            write_header = not os.path.isfile(csv_path) or (rewrite and os.path.getsize(csv_path) == 0)
            df_row = pd.DataFrame([row], columns=columns)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                df_row.to_csv(f, header=write_header, index=False)

        for image in os.listdir(f"{root_directory}/{xai}/{xai}{P}"):
            for P in P_values:
                for xai in xai_methods:
                    file_path = os.path.join(root_directory, xai, f"{xai}{P}", image)
                    if not os.path.isfile(file_path):
                        continue  # not present for this combo

                    # Parse filename like "012_goldfish.png"
                    basename = os.path.basename(file_path)
                    try:
                        # index_part, rest = basename.split('_', 1)
                        # true_label_raw = rest.split('.')[0]
                        true_label_raw = basename.split('_', 1)[1].split('.')[0]  # Extracts 'tench'
                        index_part = basename.split('_')[0]  # Extracts the index from the file name
                    except ValueError:
                        continue  # unexpected format, skip
                    true_label = true_label_raw.replace('_', ' ')

                    prompt = f"What do you see in the picture? Is it a {true_label} from the imagenet database?"

                    response = self.strategy.generate_response(
                        prompt=prompt,
                        background=self.background,
                        image=file_path,
                        jsonDescription=self.jsonDescription
                    )
                    response_values = list(response.model_dump().values())
                    correctly = response_values[0] if response_values else False

                    row = [index_part, true_label, correctly] + response_values
                    append_row(xai, P, row)


    def boolean_outputs_classification_reverse2(self, root_directory: str, save_path: str, rewrite=False):
        """
        Stream-style processing with loop order: image -> P -> xai.
        Only path pattern: {root_directory}/{xai}/{xai}{P}/{image}.
        Each image is saved immediately per (xai,P) CSV.
        Resumes by skipping already-recorded (Index, True_Label) pairs.
        """
        os.makedirs(save_path, exist_ok=True)

        xai_methods = [
            "guided_backprop",
            "guided_gradcam",
            "integrated_gradients",
            "inputxgradient",
            "smoothgrad",
        ]
        P_values = [ "05", "10", "15", "20", "25", "30", "35", "40", "45", "50",
                    "55", "60", "65", "70", "75", "80", "85", "90", "95"]

        # Cache of already seen (Index, True_Label) per combo to avoid re-reading every time
        seen_records: dict[tuple[str, str], set[tuple[str, str]]] = {}

        def load_seen(xai: str, P: str) -> set[tuple[str, str]]:
            key = (xai, P)
            if key in seen_records:
                return seen_records[key]
            xai_folder = os.path.join(save_path, xai)
            combo = f"{xai}_{P}"
            csv_path = os.path.join(xai_folder, f"{combo}.csv")
            seen = set()
            if os.path.isfile(csv_path) and not rewrite:
                try:
                    df_existing = pd.read_csv(csv_path)
                    if 'Index' in df_existing.columns and 'True_Label' in df_existing.columns:
                        for _, row in df_existing.iterrows():
                            seen.add((str(row['Index']), str(row['True_Label'])))
                except Exception:
                    pass  # if reading fails, treat as empty
            seen_records[key] = seen
            return seen

        def append_row(xai: str, P: str, row: list):
            combo = f"{xai}_{P}"
            xai_folder = os.path.join(save_path, xai)
            os.makedirs(xai_folder, exist_ok=True)
            csv_path = os.path.join(xai_folder, f"{combo}.csv")

            # If rewrite requested and file exists, remove it once so header will be written fresh
            if rewrite and os.path.isfile(csv_path):
                try:
                    os.remove(csv_path)
                except OSError:
                    pass

            llm_count = max(1, len(row) - 3)
            llm_columns = [f'llm_{i+1}' for i in range(llm_count)]
            columns = ['Index', 'True_Label', 'Match'] + llm_columns

            write_header = not os.path.isfile(csv_path)
            df_row = pd.DataFrame([row], columns=columns)
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                df_row.to_csv(f, header=write_header, index=False)

            # Update cache so we don't repeat
            index_part, true_label = row[0], row[1]
            load_seen(xai, P).add((str(index_part), str(true_label)))

        # You had this as your image source
        xai = "guided_gradcam"
        P = "05"


        for image in tqdm(os.listdir(f"{root_directory}/{xai}/{xai}{P}"), desc=f"Processing images"):
            if not image.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            for P in P_values:
                for xai in xai_methods:
                    file_path = os.path.join(root_directory, xai, f"{xai}{P}", image)

                    sys.stdout.write("\r" + f"{file_path}" + " " * 10)
                    sys.stdout.flush()

                    if not os.path.isfile(file_path):
                        continue  # file not present

                    basename = os.path.basename(file_path)
                    try:
                        index_part = basename.split('_')[0]
                        true_label_raw = basename.split('_', 1)[1].split('.')[0]
                    except (IndexError, ValueError):
                        continue  # malformed name
                    true_label = true_label_raw.replace('_', ' ')

                    # Resume check
                    seen = load_seen(xai, P)
                    if (index_part, true_label) in seen:
                        continue  # already processed

                    prompt = f"What do you see in the picture? Is it a {true_label} from the imagenet database?"

                    response = self.strategy.generate_response(
                        prompt=prompt,
                        background=self.background,
                        image=file_path,
                        jsonDescription=self.jsonDescription
                    )
                    response_values = list(response.model_dump().values())
                    correctly = response_values[0] if response_values else False

                    row = [index_part, true_label, correctly] + response_values
                    append_row(xai, P, row)
