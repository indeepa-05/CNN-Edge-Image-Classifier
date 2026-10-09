from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

class OCTMNISTDataPreprocessor:
    """
    Handles loading and preparation of the OCTMNIST dataset.
    Given the .npz file, this class will load the data, combine the original
    train/validation/test splits, and create a new stratified 70/15/15
    """
    def __init__(self, data_path, random_state=42):
        self.data_path = Path(data_path)
        self.random_state = random_state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def load_dataset(self, val_size=0.15, test_size=0.15):
        print(f"Loading OCTMNIST data...")
        data = np.load(self.data_path)
        # Load original MedMNIST splits
        X_train = data["train_images"]
        y_train = data["train_labels"]
        X_val = data["val_images"]
        y_val = data["val_labels"]
        X_test = data["test_images"]
        y_test = data["test_labels"]
        print(f"Original splits loaded: Training Datapoints: {X_train.shape[0]}, Validation Datapoints: {X_val.shape[0]}, Test Datapoints: {X_test.shape[0]}")
        # Combine the original splits
        X_combined = np.concatenate((X_train, X_val, X_test), axis=0)
        y_combined = np.concatenate((y_train, y_val, y_test), axis=0).ravel() #Flattening the labels to 1D array
        print(f"Total Data Points: {X_combined.shape[0]}")
        # Create new stratified splits
        print(f"Creating new stratified splits...")
        temp_size= val_size + test_size
        relative_test_size = test_size / temp_size
        X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=temp_size, random_state=42, stratify=y_combined)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_test_size, random_state=42,stratify=y_temp)
        print(f"New splits created: Training Datapoints: {X_train.shape[0]}, Validation Datapoints: {X_val.shape[0]}, Test Datapoints: {X_test.shape[0]}")
        # Adding New axis as Tf expects 4D input for Conv2D layers
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        print(f"Shapes of returned Splits: X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"Shapes of returned Labels: y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test
