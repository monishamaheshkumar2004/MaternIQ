import os
import numpy as np
import pandas as pd

# Define Paths
npy_folder = "C:/Users/Pradeepa/Desktop/EFM/efm1/models/"  # Folder containing .npy files
csv_folder = "C:/Users/Pradeepa/Desktop/EFM/efm1/processed/csv_data/"  # Folder to save CSVs

#  Create the CSV folder if it doesn't exist
os.makedirs(csv_folder, exist_ok=True)

#  List of structured .npy files to convert
npy_files = [
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\extracted_features.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\features.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\features_balanced.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\labels_balanced.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\labels.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\selected_features.npy",
    r"C:\Users\Pradeepa\Desktop\ehg_preterm\selected_feature_indices.npy"
]

#  Convert each .npy file to CSV
for file_name in npy_files:
    npy_path = os.path.join(npy_folder, file_name)
    csv_path = os.path.join(csv_folder, file_name.replace(".npy", ".csv"))

    try:
        data = np.load(npy_path, allow_pickle=True)  # Load .npy file
        df = pd.DataFrame(data)  # Convert to DataFrame
        df.to_csv(csv_path, index=False)  # Save as CSV
        print(f" Converted {file_name} to {csv_path}")
    except Exception as e:
        print(f" Error processing {file_name}: {e}")

print("🎯 All structured .npy files have been converted to CSV and saved!")
