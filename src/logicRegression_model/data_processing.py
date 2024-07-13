import pandas as pd

# Load and clean the data from a CSV file
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Print the column names of the CSV file
    print("Column names in the CSV file:")
    print(df.columns)

    # Drop rows with NaN values in the 'Average Mark var' column
    df_cleaned = df.dropna(subset=['Average Mark var'])

    # Check for NaN values
    print("Checking for NaN values:")
    print(df_cleaned.isna().sum())

    # Check for infinity values
    print("Checking for infinity values:")
    print(df_cleaned.applymap(lambda x: x == float('inf')).sum())

    # Drop remaining NaN values
    df_cleaned = df_cleaned.dropna()
    # Remove rows with infinity values
    df_cleaned = df_cleaned[~df_cleaned.isin([float('inf'), float('-inf')]).any(1)]

    return df_cleaned
