import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

# Load dataset from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare features (X) and target (y) from the dataset
def prepare_data(data):
    X = data[['age_at_event mean', 'Average Net Result mean', 'Mark mean',
              'Mark min', 'Mark var', 'current_match_ranking mean',
              'current_match_ranking min', 'Average Rank mean', 'efficiency mean', 'Nat', 'Years to Olympics']]
    y = data['Olympic Ranking']
    return X, y

# Balance classes using ADASYN
def oversample_data(X, y):
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled

# Split dataset into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
