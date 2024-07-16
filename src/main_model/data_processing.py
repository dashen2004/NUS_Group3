import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load dataset from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)


def transform_to_binary(series):
    return series.apply(lambda x: 1 if 1 <= x <= 8 else 0)


# Prepare features (X) and target (y) from the dataset
def prepare_data(data):
    X = data[['age_at_event mean', 'Average Net Result mean', 'Mark mean',
              'Mark min', 'Mark var', 'current_match_ranking mean',
              'current_match_ranking min', 'Average Rank mean', 'efficiency mean', 'Nat', 'Years to Olympics']]
    y = data['Olympic Ranking']
    return X, y


def expand_data(test, times=5):
    expanded_df = pd.concat([test] * times, ignore_index=True)
    return expanded_df


# Balance classes using ADASYN
def oversample_data(X, y):
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled


# Split dataset into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=42)


# Noise Injection
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


# Feature Perturbation
def perturb_features(X, perturb_fraction=0.1):
    X_perturbed = X.copy()
    n_samples, n_features = X.shape
    n_perturb = int(n_samples * perturb_fraction)
    for _ in range(n_perturb):
        sample_idx = np.random.randint(n_samples)
        feature_idx = np.random.randint(n_features)
        perturb_value = np.random.normal(0, 0.1)
        X_perturbed.iloc[sample_idx, feature_idx] += perturb_value
    return X_perturbed


# Cluster-Based Synthetic Data Generation
def generate_synthetic_data(X, n_clusters=10, n_samples_per_cluster=100):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)

    synthetic_data = []
    for cluster_center in kmeans.cluster_centers_:
        for _ in range(n_samples_per_cluster):
            sample = cluster_center + np.random.normal(0, 0.1, X_scaled.shape[1])
            synthetic_data.append(sample)

    synthetic_data = scaler.inverse_transform(synthetic_data)
    return pd.DataFrame(synthetic_data, columns=X.columns)


# Data Augmentation Pipeline
def augment_data(X, y, target_size=8000):
    current_size = X.shape[0]
    augmented_data = X.copy()
    augmented_labels = y.copy()

    while current_size < target_size:
        X_noisy = add_noise(X)
        X_perturbed = perturb_features(X)
        X_synthetic = generate_synthetic_data(X)

        new_data = pd.concat([X_noisy, X_perturbed, X_synthetic])
        new_labels = pd.concat([y, y, y])

        augmented_data = pd.concat([augmented_data, new_data])
        augmented_labels = pd.concat([augmented_labels, new_labels])

        current_size = augmented_data.shape[0]

        augmented_data, augmented_labels = augmented_data.drop_duplicates(), augmented_labels.loc[augmented_data.index]

    if augmented_data.shape[0] > target_size:
        indices = np.random.choice(augmented_data.index, size=target_size, replace=False)
        augmented_data = augmented_data.loc[indices]
        augmented_labels = augmented_labels.loc[indices]

    return augmented_data, augmented_labels

