import pandas as pd
import os


def preprocess_knapsack_data(folder="raw/", save_folder="processed/"):
    """
    Preprocess knapsack instance files by normalizing weights and values,
    and saving the preprocessed files.

    Parameters:
    - folder (str): Directory containing raw knapsack instances (CSV format).
    - save_folder (str): Directory to save the processed instances.

    Returns:
    - None
    """
    # Create the processed folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Process each file in the folder
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path)

            # Normalize weights and values (scaling between 0 and 1)
            df['weight'] = df['weight'] / df['weight'].max()
            df['value'] = df['value'] / df['value'].max()

            # Save the preprocessed data
            save_path = os.path.join(save_folder, filename)
            df.to_csv(save_path, index=False)
            print(f"Saved preprocessed data to {save_path}")


# Example of how to use the preprocessing function
if __name__ == "__main__":
    preprocess_knapsack_data()
