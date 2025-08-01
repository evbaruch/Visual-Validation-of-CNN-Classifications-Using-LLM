import os
import pandas as pd
import matplotlib.pyplot as plt
# from dataset_API.dataset_interface import dataset_interface
import re



# def calculate_accuracy(root_directory: str, save_path: str):
#     """
#     Calculates the average 'Match' value for each CSV file in the root directory 
#     and saves the results to a specified CSV file.

#     Args:
#         root_directory (str): The directory to search for CSV files.
#         save_path (str): The file path to save the output CSV.

#     Returns:
#         None
#     """
#     results = []  # List to store results
#     os.makedirs(save_path, exist_ok=True)

#     for dirpath, _, filenames in os.walk(root_directory): 
#         # Filter for CSV files
#         csv_files = [f for f in filenames if f.endswith('.csv')]

#         for file in csv_files:
#             # Get the full path of the file
#             file_path = os.path.join(dirpath, file)

#             # Load the CSV file into a DataFrame
#             df = pd.read_csv(file_path)

#             if 'Match' not in df.columns:
#                 continue
                
#             # Calculate the average of the 'Match' column
#             average_match = df['Match'].mean()

#             # Append the result to the list
#             results.append({"File": file, "Average Match": average_match})

#     # Convert results to a DataFrame and save to CSV
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(f"{save_path}\\results.csv", index=False)

#     create_results_table(f"{save_path}\\results.csv", f"{save_path}\\pivot_table.csv")
#     plot_graphs_from_table(f"{save_path}\\results.csv", save_path)
#     combined_plot_graphs_from_table(f"{save_path}\\results.csv", save_path)


# def plot_graphs_from_table(data_path: str, save_directory: str):
#     """
#     Creates graphs for each combination of 'method' and 'pre_trained_model'.
#     The x-axis is 'threshold', and the y-axis is 'Average Match'.
#     Graphs are saved as PNG files, with each line connecting two points showing the delta.

#     Args:
#         data_path (str): Path to the CSV file containing the data.
#         save_directory (str): Directory to save the graphs.

#     Returns:
#         None
#     """
#     # Load the data from the CSV file
#     data = pd.read_csv(data_path)

#     # Extract 'method', 'threshold', and 'pre_trained_model' from the 'File' column using parse_file_name
#     data[['method', 'threshold', 'pre_trained_model']] = data['File'].apply(
#         lambda x: pd.Series(dataset_interface.parse_file_name(x))
#     )

#     # Convert 'threshold' to numeric for sorting and plotting
#     data['threshold'] = pd.to_numeric(data['threshold'])

#     # Ensure the save directory exists
#     os.makedirs(save_directory, exist_ok=True)

#     # Group by 'method' and 'pre_trained_model'
#     grouped = data.groupby(['method', 'pre_trained_model'])

#     for (method, pre_trained_model), group in grouped:
#         plt.figure(figsize=(8, 6))

#         # Sort by threshold
#         group = group.sort_values('threshold')

#         # Plot the data
#         plt.plot(group['threshold'], group['Average Match'], marker='o', label=pre_trained_model)
        
#         # Add deltas and the average delta
#         deltas = group['Average Match'].diff().iloc[1:]  # Get the difference between consecutive points
#         for i in range(1, len(group)):
#             delta = deltas.iloc[i - 1]
#             # Display delta between points
#             plt.text(group['threshold'].iloc[i], group['Average Match'].iloc[i],
#                      f'{delta:.2f}', fontsize=9, color='red', ha='center', va='bottom')

#         # Calculate and display the average delta
#         average_delta = deltas.mean()
#         plt.text(0.95, 0.85, f'Avg Delta: {average_delta:.2f}', fontsize=12, transform=plt.gca().transAxes,
#                  ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

#         plt.title(f'Method: {method}, Model: {pre_trained_model}')
#         plt.xlabel('Threshold')
#         plt.ylabel('Average Match')
#         #plt.xscale('log')
#         plt.grid(True)
#         plt.legend()

#         # Save the plot
#         file_name = f"{method}_{pre_trained_model}.png".replace(' ', '_')
#         save_path = os.path.join(save_directory, file_name)
#         plt.savefig(save_path)
#         plt.close()

#         print(f"Saved plot: {save_path}")


# def combined_plot_graphs_from_table(data_path: str, save_directory: str):
#     """
#     Creates a combined graph for each combination of 'method' and 'pre_trained_model'.
#     The x-axis is 'threshold', and the y-axis is 'Average Match'.
#     All graphs are drawn on the same plot and saved as a single PNG file.

#     Args:
#         data_path (str): Path to the CSV file containing the data.
#         save_directory (str): Directory to save the combined graph.

#     Returns:
#         None
#     """
#     # Load the data from the CSV file
#     data = pd.read_csv(data_path)

#     # Extract 'method', 'threshold', and 'pre_trained_model' from the 'File' column using parse_file_name
#     data[['method', 'threshold', 'pre_trained_model']] = data['File'].apply(
#         lambda x: pd.Series(dataset_interface.parse_file_name(x))
#     )

#     # Convert 'threshold' to numeric for sorting and plotting
#     data['threshold'] = pd.to_numeric(data['threshold'])

#     # Ensure the save directory exists
#     os.makedirs(save_directory, exist_ok=True)

#     # Create a figure for the combined graph
#     plt.figure(figsize=(10, 8))

#     # Group by 'method' and 'pre_trained_model'
#     grouped = data.groupby(['method', 'pre_trained_model'])

#     for (method, pre_trained_model), group in grouped:
#         # Sort by threshold
#         group = group.sort_values('threshold')

#         # Plot the data for each combination
#         plt.plot(group['threshold'], group['Average Match'], marker='o', label=f'{method} - {pre_trained_model}')

#     # Set the title, labels, and scale
#     plt.title('Threshold vs. Average Match for All Methods and Models')
#     plt.xlabel('Threshold')
#     plt.ylabel('Average Match')
#     # plt.xscale('log')  # Apply log scale to the x-axis
#     plt.grid(True)
#     plt.legend()

#     # Save the combined plot
#     save_path = os.path.join(save_directory, "combined_plot.png")
#     plt.savefig(save_path)
#     plt.close()

#     print(f"Saved combined plot: {save_path}")



# def create_results_table(original_file, save_location):
#     """
#     Create a pivot table from the original file where rows are methods,
#     columns are percentages, and values are average match scores.
#     Supports any prefix before the percentage and method in the filename.
    
#     :param original_file: Path to the input CSV file.
#     :param save_location: Path to save the output CSV file.
#     """
#     # Read the original CSV
#     df = pd.read_csv(original_file)
    
#     # Use a general regex pattern to extract the first number and then the method
#     df['Percentage'] = df['File'].apply(lambda x: int(re.search(r'_(\d+)_', x).group(1)))
#     df['Method'] = df['File'].apply(lambda x: re.search(r'_\d+_(.+)\.csv', x).group(1))
    
#     # df['Percentage'] = df['File'].apply(lambda x: int(re.search(r'_(\d+)', x).group(1)))
#     # df['Method'] = original_file.split("\\")[-2].replace(".csv", "")
    
#     # Create the pivot table
#     pivot = df.pivot(index='Method', columns='Percentage', values='Average Match')
    
#     # Sort the columns (percentages) in ascending order
#     pivot = pivot[sorted(pivot.columns)]
    
#     # Save to CSV
#     pivot.to_csv(save_location)
    
#     print(f"Pivot table created and saved to {save_location}")

# def inverted_tables(data_path: str, save_directory: str):
#     """
#     Creates inverted tables for each method (resnet18, v3_inception, v3_large, v3_small).
#     Rows will be the file names (list_of_directories), and columns will be percentages.

#     Args:
#         data_path (str): Path to the directory containing subdirectories with pivot tables.
#         save_directory (str): Path to save the inverted tables.

#     Returns:
#         None
#     """
#     os.makedirs(save_directory, exist_ok=True)  # Ensure the save directory exists

#     # Initialize dictionaries to store data for each method
#     methods_data = {
#         "resnet18": {},
#         "v3_inception": {},
#         "v3_large": {},
#         "v3_small": {}
#     }

#     list_of_directories = os.listdir(data_path)
#     for directory in list_of_directories:
#         path = os.path.join(data_path, f"{directory}\\boolean\\results") # <= TODO: Change this to the correct path
#         pivot_file = os.path.join(path, "pivot_table.csv")
        
#         if os.path.exists(pivot_file):
#             # Read the pivot table
#             pivot_table = pd.read_csv(pivot_file, index_col=0)
            
#             # Add data for each method
#             for method in methods_data.keys():
#                 if method in pivot_table.index:
#                     methods_data[method][directory] = pivot_table.loc[method].to_dict()

#     # Create and save inverted tables for each method
#     for method, data in methods_data.items():
#         if data:  # Only process if there is data for the method
#             # Convert the dictionary to a DataFrame
#             inverted_table = pd.DataFrame.from_dict(data, orient="index")
            
#             # Save the inverted table to a CSV file
#             save_path = os.path.join(save_directory, f"{method}_inverted_table.csv")
#             inverted_table.to_csv(save_path)
#             create_graph_from_table(save_path, os.path.join(save_directory, f"{method}_graph.png"))
#             print(f"Inverted table for {method} saved to {save_path}")

# def create_graph_from_table(table_path: str, save_path: str):
#     """
#     Creates a graph from a single inverted table and saves it as an image.

#     Args:
#         table_path (str): Path to the CSV file containing the inverted table.
#         save_path (str): Path to save the generated graph.

#     Returns:
#         None
#     """
#     # Read the inverted table
#     table = pd.read_csv(table_path, index_col=0)

#     # Plot the data
#     plt.figure(figsize=(10, 6))
#     for index, row in table.iterrows():
#         plt.plot(row.index, row.values, marker='o', label=index)

#     # Add title, labels, and legend
#     method_name = table_path.split("\\")[-1].replace("_inverted_table.csv", "")
#     plt.title(f"Performance Graph for {method_name}", fontsize=14)
#     plt.xlabel("Percentage", fontsize=12)
#     plt.ylabel("Performance", fontsize=12)
#     plt.legend(title="Directories", fontsize=10)
#     plt.grid(True)

#     # Save the graph as an image
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Graph saved to {save_path}")           
                
# # Example usage
# # inverted_tables("data\\midPrecentage", "data\\midPrecentage\\inverted_tables")



def plot_success_rate(save_path: str):
    """
    Reads all per-(xai,P) CSVs under save_path/<xai>/ and plots success rate (mean of Match)
    vs P for each xai method. Assumes filenames like: guided_gradcam_05.csv
    """
    # Regex to extract xai and P from filename
    filename_re = re.compile(r'^(?P<xai>.+)_(?P<P>\d{2})\.csv$')
    # filename_re = re.compile(r'^(?P<method>.+?)_(?P<P>\d{2})(?:_.*)?\.csv$', re.IGNORECASE)
    
    # Data accumulation: {xai: {P_int: success_rate}}
    results: dict[str, dict[int, float]] = {}

    for xai in sorted(os.listdir(save_path)):
        xai_dir = os.path.join(save_path, xai)
        if not os.path.isdir(xai_dir):
            continue
        for fname in sorted(os.listdir(xai_dir)):
            m = filename_re.match(fname)
            if not m:
                continue
            xai_name = m.group('xai')
            P_str = m.group('P')
            try:
                P_val = int(P_str)
            except ValueError:
                continue

            csv_path = os.path.join(xai_dir, fname)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue  # skip unreadable files

            if 'Match' not in df.columns:
                continue  # unexpected format

            # Normalize Match to boolean/0-1
            match_series = df['Match']
            # If it's strings like 'True'/'False', convert
            if match_series.dtype == object:
                match_series = match_series.map(lambda x: str(x).lower() in ('true', '1', 't', 'yes'))

            try:
                success_rate = float(match_series.mean())
            except Exception:
                continue

            results.setdefault(xai_name, {})[P_val] = success_rate

    if not results:
        print("No data found to plot.")
        return

    # Build a DataFrame with index = sorted P values, columns = xai methods
    all_Ps = sorted({p for sub in results.values() for p in sub.keys()})
    df_plot = pd.DataFrame(index=all_Ps)

    for xai_name, perP in results.items():
        series = pd.Series({p: perP.get(p, float('nan')) for p in all_Ps})
        df_plot[xai_name] = series

    # Plot
    plt.figure()
    for xai_name in sorted(df_plot.columns):
        plt.plot(df_plot.index, df_plot[xai_name], marker='o', label=xai_name)

    plt.xlabel("P")
    plt.ylabel("Success Rate (mean of Match)")
    plt.title("Success Rate vs P by XAI Method")
    plt.xticks(all_Ps, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# plot_success_rate("data/llm_answer_Avi_grey")

def plot_success_rate_and_dump_table_swapped(base_save_path: str):
    """
    Aggregates success rate (mean of Match) per method and P, saves both normal and transposed tables,
    and plots:
      - P vs success rate (standard)
      - success rate vs P (axes swapped)
    """
    filename_re = re.compile(r'^(?P<method>.+?)_(?P<P>\d{2})(?:_.*)?\.csv$', re.IGNORECASE)
    accumulation: dict[str, dict[int, list[float]]] = {}

    for method_dirname in sorted(os.listdir(base_save_path)):
        method_dir = os.path.join(base_save_path, method_dirname)
        if not os.path.isdir(method_dir):
            continue
        for fname in sorted(os.listdir(method_dir)):
            if not fname.lower().endswith('.csv'):
                continue
            m = filename_re.match(fname)
            if not m:
                continue
            method = m.group('method')
            P_str = m.group('P')
            try:
                P_val = int(P_str)
            except ValueError:
                continue

            csv_path = os.path.join(method_dir, fname)
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'Match' not in df.columns:
                continue

            match_series = df['Match']
            if match_series.dtype == object:
                match_bool = match_series.map(lambda x: str(x).strip().lower() in ('true', '1', 't', 'yes', 'y'))
                match_numeric = match_bool.astype(float)
            else:
                match_numeric = pd.to_numeric(match_series, errors='coerce')
                if match_series.dtype == bool:
                    match_numeric = match_series.astype(float)

            if match_numeric.dropna().empty:
                continue
            success_rate = float(match_numeric.mean())
            accumulation.setdefault(method, {}).setdefault(P_val, []).append(success_rate)

    if not accumulation:
        print("No data found.")
        return

    all_Ps = sorted({p for per_method in accumulation.values() for p in per_method.keys()})
    summary_df = pd.DataFrame(index=all_Ps)

    for method, perP in accumulation.items():
        averaged = {p: (sum(perP[p]) / len(perP[p])) for p in perP}
        series = pd.Series({p: averaged.get(p, float('nan')) for p in all_Ps})
        summary_df[method] = series

    # Save standard and transposed tables
    out_csv = os.path.join(base_save_path, "success_rate_summary.csv")
    summary_df.to_csv(out_csv, index_label="P")
    out_csv_t = os.path.join(base_save_path, "success_rate_summary_transposed.csv")
    summary_df.T.to_csv(out_csv_t, index_label="Method")
    print(f"Saved summary: {out_csv}")
    print(f"Saved transposed summary: {out_csv_t}")

    # Plot
    plt.figure()
    for xai_name in sorted(summary_df.columns):
        plt.plot(summary_df.index, summary_df[xai_name], marker='o', label=xai_name)

    plt.xlabel("P")
    plt.ylabel("Success Rate (mean of Match)")
    plt.title("Success Rate vs P by XAI Method")
    plt.xticks(all_Ps, rotation=45)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_success_rate_and_dump_table_swapped("data/llm_answer_Avi_grey")