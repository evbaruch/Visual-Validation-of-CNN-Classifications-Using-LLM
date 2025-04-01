import os
import pandas as pd
import matplotlib.pyplot as plt
from dataset_API.dataset_interface import dataset_interface

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
    Calculates the average 'Match' value for each CSV file in the root directory 
    and saves the results to a specified CSV file.

    Args:
        root_directory (str): The directory to search for CSV files.
        save_path (str): The file path to save the output CSV.

    Returns:
        None
    """
    results = []  # List to store results
    os.makedirs(save_path, exist_ok=True)

    for dirpath, _, filenames in os.walk(root_directory): 
        # Filter for CSV files
        csv_files = [f for f in filenames if f.endswith('.csv')]

        for file in csv_files:
            # Get the full path of the file
            file_path = os.path.join(dirpath, file)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            if 'Match' not in df.columns:
                continue
                
            # Calculate the average of the 'Match' column
            average_match = df['Match'].mean()

            # Append the result to the list
            results.append({"File": file, "Average Match": average_match})

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{save_path}\\results.csv", index=False)

    plot_graphs_from_table(f"{save_path}\\results.csv", save_path)
    combined_plot_graphs_from_table(f"{save_path}\\results.csv", save_path)


def plot_graphs_from_table(data_path: str, save_directory: str):
    """
    Creates graphs for each combination of 'method' and 'pre_trained_model'.
    The x-axis is 'threshold', and the y-axis is 'Average Match'.
    Graphs are saved as PNG files, with each line connecting two points showing the delta.

    Args:
        data_path (str): Path to the CSV file containing the data.
        save_directory (str): Directory to save the graphs.

    Returns:
        None
    """
    # Load the data from the CSV file
    data = pd.read_csv(data_path)

    # Extract 'method', 'threshold', and 'pre_trained_model' from the 'File' column using parse_file_name
    data[['method', 'threshold', 'pre_trained_model']] = data['File'].apply(
        lambda x: pd.Series(dataset_interface.parse_file_name(x))
    )

    # Convert 'threshold' to numeric for sorting and plotting
    data['threshold'] = pd.to_numeric(data['threshold'])

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Group by 'method' and 'pre_trained_model'
    grouped = data.groupby(['method', 'pre_trained_model'])

    for (method, pre_trained_model), group in grouped:
        plt.figure(figsize=(8, 6))

        # Sort by threshold
        group = group.sort_values('threshold')

        # Plot the data
        plt.plot(group['threshold'], group['Average Match'], marker='o', label=pre_trained_model)
        
        # Add deltas and the average delta
        deltas = group['Average Match'].diff().iloc[1:]  # Get the difference between consecutive points
        for i in range(1, len(group)):
            delta = deltas.iloc[i - 1]
            # Display delta between points
            plt.text(group['threshold'].iloc[i], group['Average Match'].iloc[i],
                     f'{delta:.2f}', fontsize=9, color='red', ha='center', va='bottom')

        # Calculate and display the average delta
        average_delta = deltas.mean()
        plt.text(0.95, 0.85, f'Avg Delta: {average_delta:.2f}', fontsize=12, transform=plt.gca().transAxes,
                 ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        plt.title(f'Method: {method}, Model: {pre_trained_model}')
        plt.xlabel('Threshold')
        plt.ylabel('Average Match')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()

        # Save the plot
        file_name = f"{method}_{pre_trained_model}.png".replace(' ', '_')
        save_path = os.path.join(save_directory, file_name)
        plt.savefig(save_path)
        plt.close()

        print(f"Saved plot: {save_path}")


def combined_plot_graphs_from_table(data_path: str, save_directory: str):
    """
    Creates a combined graph for each combination of 'method' and 'pre_trained_model'.
    The x-axis is 'threshold', and the y-axis is 'Average Match'.
    All graphs are drawn on the same plot and saved as a single PNG file.

    Args:
        data_path (str): Path to the CSV file containing the data.
        save_directory (str): Directory to save the combined graph.

    Returns:
        None
    """
    # Load the data from the CSV file
    data = pd.read_csv(data_path)

    # Extract 'method', 'threshold', and 'pre_trained_model' from the 'File' column using parse_file_name
    data[['method', 'threshold', 'pre_trained_model']] = data['File'].apply(
        lambda x: pd.Series(dataset_interface.parse_file_name(x))
    )

    # Convert 'threshold' to numeric for sorting and plotting
    data['threshold'] = pd.to_numeric(data['threshold'])

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Create a figure for the combined graph
    plt.figure(figsize=(10, 8))

    # Group by 'method' and 'pre_trained_model'
    grouped = data.groupby(['method', 'pre_trained_model'])

    for (method, pre_trained_model), group in grouped:
        # Sort by threshold
        group = group.sort_values('threshold')

        # Plot the data for each combination
        plt.plot(group['threshold'], group['Average Match'], marker='o', label=f'{method} - {pre_trained_model}')

    # Set the title, labels, and scale
    plt.title('Threshold vs. Average Match for All Methods and Models')
    plt.xlabel('Threshold')
    plt.ylabel('Average Match')
    plt.xscale('log')  # Apply log scale to the x-axis
    plt.grid(True)
    plt.legend()

    # Save the combined plot
    save_path = os.path.join(save_directory, "combined_plot.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved combined plot: {save_path}")