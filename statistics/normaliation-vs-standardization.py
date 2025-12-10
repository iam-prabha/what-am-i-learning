# %%
import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# List files in the data directory (similar to original script)
def list_data_files(data_dir="../data"):
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def load_data(path="../data/fish_dataset.csv"):
    """Load the fish dataset from CSV."""
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return data


def standardize_series(series):
    """Standardize a pandas Series and return a 1D numpy array."""
    # convert to numpy array first to avoid ExtensionArray issues, then reshape
    arr = series.to_numpy().reshape(-1, 1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(arr).ravel()
    return scaled


def normalize_series(series):
    """Min-max scale a pandas Series and return a 1D numpy array."""
    arr = series.to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr).ravel()
    return scaled


def main():
    list_data_files()

    data = load_data()
    print("\nData head:")
    print(data.head())

    # Use the "Fish count" column; ensure it exists
    col = "Fish count"
    if col not in data.columns:
        raise KeyError(
            f"Column '{col}' not found in dataset. Available columns: {list(data.columns)}"
        )

    print("\nOriginal describe():")
    print(data.describe().round(2))

    # Standardization
    scaled_std = standardize_series(data[col])
    # fit_transform returns a 2D array; assign a flattened column or keep as 2D
    data["scaled fish count"] = scaled_std

    print("\nAfter standardization describe():")
    print(data.describe().round(2))

    # Normalization (Min-Max)
    scaled_mm = normalize_series(data[col])
    data["scaled fish count2"] = scaled_mm

    print("\nAfter normalization describe():")
    print(data.describe().round(2))

    # Show a sample of the new columns
    print("\nSample with scaled columns:")
    print(data[[col, "scaled fish count", "scaled fish count2"]].head())


if __name__ == "__main__":
    main()
