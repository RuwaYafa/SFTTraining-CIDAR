
"""
In command line apply// try to put them in requirements.txt
PS E:\SFTTraining-CIDAR> pip install pandas scikit-learn
PS E:\SFTTraining-CIDAR> python data\splitdata.py
"""

import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split

def split_cidar_dataset(file_path, test_size=0.2, random_state=42):
    """
    Splits the cidar-train-data-display.csv dataset into training and testing sets,
    using only the 'instruction' and 'output' columns.

    Args:
        file_path (str): The path to the CSV file.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional):  Controls the shuffling applied to the data
                                       before applying the split.  Defaults to 42.

    Returns:
        tuple: A tuple containing the training and testing sets as pandas DataFrames,
               each with only the 'instruction' and 'output' columns.
               Returns (None, None) if the file is not found or an error occurs.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None, None

    # Check if the required columns exist
    if not all(col in df.columns for col in ['instruction', 'output']):
        print("Error: 'instruction' or 'output' column not found in the dataset.")
        return None, None

    # Extract the 'instruction' and 'output' columns
    data = df[['instruction', 'output']]

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    print(f"Dataset successfully split. Training set size: {len(train_data)}, Testing set size: {len(test_data)}")
    return train_data, test_data

if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = r"E:\SFTTraining-CIDAR\data\cidar-train-data-display.csv"  # Use a raw string

    # Split the dataset
    train_df, test_df = split_cidar_dataset(file_path)

    # Check if the dataframes are valid before proceeding.
    if train_df is not None and test_df is not None:
        # Save the training and testing sets to new CSV files (optional)
        train_df.to_csv("e:/SFTTraining-CIDAR/data/train_data.csv", index=False)
        test_df.to_csv("e:/SFTTraining-CIDAR/data/test_data.csv", index=False)

        # Display the first 5 rows of the training and testing sets
        print("\nFirst 5 rows of the training set:")
        print(train_df.head())
        print("\nFirst 5 rows of the testing set:")
        print(test_df.head())
