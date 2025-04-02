import os
import pandas as pd
import matplotlib.pyplot as plt

# use example : add_precentage_to_csv("GradientShap", "boolean", "data\\llm_answer\\GradientShap\\boolean\\results")
def add_precentage_to_csv(method: str, query_type: str, save_directory: str):
  
    # Build the path to the directory containing the files
    root_directory = os.path.join("data", "llm_answer", method, query_type)

    # Get the path to the directory containing the percentage files
    percentage_directory = os.path.join("data", "mid", method, "csv")

    # do a walk in the directories and for each 2 files with the same name in the two directories
    # do a join 
    # from the percentage file get the percentage and add it to the csv file
    # and from the root file get all the columns and add them to the csv file
    # and save the new csv file in the root directory with the same name as the original file

    for dirpath, _, filenames in os.walk(root_directory):
        # Filter for CSV files
        csv_files = [f for f in filenames if f.endswith('.csv')]

        for file in csv_files:
            # Get the full path of the file
            file_path = os.path.join(dirpath, file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Get the name of the file without the extension
            file_name = os.path.splitext(file)[0]

            # Build the path to the percentage file
            percentage_file_path = os.path.join(percentage_directory, f"{file_name}.csv")

            # Load the percentage file into a DataFrame
            percentage_df = pd.read_csv(percentage_file_path)

            # Merge the two DataFrames on 'File' column
            df["Removed_Percentage"] = percentage_df["Removed_Percentage"].values

            # Save the merged DataFrame to a new CSV file in the root directory
            df.to_csv(file_path, index=False)


def calculate_accuracy(root_directory: str, save_path: str):
    """
    Groups the data by 'pre_trained_model' and processes CSV files.
    """
    results = []  # List to store results
    fileNameStack = {}  # Dictionary to store file names for each pre-trained model
    os.makedirs(save_path, exist_ok=True)

    # Group the data by 'pre_trained_model'
    for dirpath, _, filenames in os.walk(root_directory):
        # Filter for CSV files
        csv_files = [f for f in filenames if f.endswith('.csv')]

        for file in csv_files:
            pre_trained_model = file.split("_")[-1].split(".")[0]  # Extract pre-trained model name from the file name
            if pre_trained_model in fileNameStack:
                # If the pre-trained model already exists in the stack, append the file to its list
                fileNameStack[pre_trained_model].append(file)
            else:
                # Create a new entry for the pre-trained model
                fileNameStack[pre_trained_model] = [file]

    pre_trained_model_matrix = {}  # List to store the pre-trained model matrix

    for pre_trained_model, files in fileNameStack.items():
        for file in files:
            # Get the full path of the file
            file_path = os.path.join(root_directory, file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            if pre_trained_model not in pre_trained_model_matrix:
                # Create a new entry for the pre-trained model
                pre_trained_model_matrix[pre_trained_model] = df
            else:
                # Merge the DataFrames for the same pre-trained model
                pre_trained_model_matrix[pre_trained_model] = pd.concat([pre_trained_model_matrix[pre_trained_model], df])

    # sort the DataFrame by 'Removed_Percentage' 
    for pre_trained_model, df in pre_trained_model_matrix.items():
        pre_trained_model_matrix[pre_trained_model] = df.sort_values('Removed_Percentage')

    # calculate the average match for each pre-trained model in a delta of each 5 percent
    for pre_trained_model, df in pre_trained_model_matrix.items():
        for i in range(0, 100, 5):
            # Filter the DataFrame for the current percentage range
            filtered_df = df[(df['Removed_Percentage'] >= i) & (df['Removed_Percentage'] < i + 10)]

            # Calculate the average of the 'Match' column
            average_match = filtered_df['Match'].mean()

            # Append the result to the list
            results.append({"File": pre_trained_model, "Average Match": average_match, "Removed_Percentage": i})
    
    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)

    results_df.to_csv(f"{save_path}\\results.csv", index=False)

    print(f"Saved results to {save_path}\\results.csv")

    #plot graphs from table
    plot_graphs_from_table(f"{save_path}\\results.csv", save_path)

    combined_plot_graphs_from_table(f"{save_path}\\results.csv", save_path)


def plot_graphs_from_table(data_path: str, save_directory: str):
    """
    Creates graphs for each combination of 'pre_trained_model'.
    The x-axis is 'Removed_Percentage', and the y-axis is 'Average Match'.
    Graphs are saved as PNG files.

    Args:
        data_path (str): Path to the CSV file containing the data.
        save_directory (str): Directory to save the graphs.

    Returns:
        None
    """
    # Load the data from the CSV file
    data = pd.read_csv(data_path)

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Group by 'File' (pre-trained model)
    grouped = data.groupby('File')

    for pre_trained_model, group in grouped:
        plt.figure(figsize=(8, 6))

        # Plot 'Removed_Percentage' vs 'Average Match'
        plt.plot(group['Removed_Percentage'], group['Average Match'], marker='o', label=pre_trained_model, color='blue')

        plt.title(f'Pre-trained Model: {pre_trained_model}')
        plt.xlabel('Removed Percentage')
        plt.ylabel('Average Match')
        plt.grid(True)
        plt.legend()

        # Save the plot
        file_name = f"{pre_trained_model}_plot.png".replace(' ', '_')
        save_path = os.path.join(save_directory, file_name)
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot: {save_path}")

def combined_plot_graphs_from_table(data_path: str, save_directory: str):
    """
    Creates a combined graph for all 'pre_trained_model'.
    The x-axis is 'Removed_Percentage', and the y-axis is 'Average Match'.
    All graphs are drawn on the same plot and saved as a single PNG file.

    Args:
        data_path (str): Path to the CSV file containing the data.
        save_directory (str): Directory to save the combined graph.

    Returns:
        None
    """
    # Load the data from the CSV file
    data = pd.read_csv(data_path)

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Create a figure for the combined graph
    plt.figure(figsize=(10, 8))

    # Group by 'File' (pre-trained model)
    grouped = data.groupby('File')

    for pre_trained_model, group in grouped:
        # Plot 'Removed_Percentage' vs 'Average Match' for each pre-trained model
        plt.plot(group['Removed_Percentage'], group['Average Match'], marker='o', label=pre_trained_model)

    # Set the title, labels, and grid
    plt.title('Removed Percentage vs. Average Match for All Pre-trained Models')
    plt.xlabel('Removed Percentage')
    plt.ylabel('Average Match')
    plt.grid(True)
    plt.legend()

    # Save the combined plot
    save_path = os.path.join(save_directory, "combined_plot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved combined plot: {save_path}")

# calculate_accuracy("data\\llm_answer\\Saliency\\boolean", "data\\llm_answer\\Saliency\\boolean\\results_precentage")
add_precentage_to_csv("Random", "boolean", "data\\llm_answer\\Random\\boolean\\results")
calculate_accuracy("data\\llm_answer\\Random\\boolean", "data\\llm_answer\\Random\\boolean\\results_precentage")

# results.calculate_accuracy("data\\llm_answer\\Saliency\\boolean", "data\\llm_answer\\Saliency\\boolean\\results")

