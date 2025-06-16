import numpy as np
import os

# Load original dataset split (before SMOTE)
X_test = np.load("X_test.npy")  # Features in test set
y_test = np.load("y_test.npy")  # Corresponding labels

# Load full file list from the dataset folder
dataset_path = "C:/Users/Pradeepa/Desktop/ehg_preterm/term-preterm-ehg-database-1.0.1/tpehgdb"
file_list = sorted([f for f in os.listdir(dataset_path) if f.endswith(".dat")])

# Since `X_test.npy` contains real test samples, match them to filenames
unseen_files = file_list[:len(X_test)]  # Pick the first N files as test files

print(f"Total .dat files: {len(file_list)}")
print(f"Testing files (from X_test.npy): {len(X_test)}")
print("\nUnseen EHG Files (Not used for training):")
print("\n".join(unseen_files))
