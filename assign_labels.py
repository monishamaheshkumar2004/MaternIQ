import os
import numpy as np

# Set the correct folder path
folder_path = r"C:\Users\Pradeepa\Desktop\ehg_preterm\term-preterm-ehg-database-1.0.1\tpehgdb"

file_names = []
labels = []

for file in os.listdir(folder_path):
    if file.endswith(".hea"):
        file_path = os.path.join(folder_path, file)
        
        with open(file_path, "r") as f:
            lines = f.readlines()

        gestation_week = None
        for line in lines:
            if "Gestation" in line:
                words = line.split()
                for word in words:
                    try:
                        gestation_week = float(word)  # Extract decimal gestation week
                        break
                    except ValueError:
                        continue  # Skip non-numeric values
        
        # Assign label if a valid gestation week was found
        if gestation_week is not None:
            label = 0 if gestation_week >= 37 else 1
            file_names.append(file.replace(".hea", ""))
            labels.append(label)
        else:
            print(f"Skipping {file}: No valid Gestation week found")

# Convert to NumPy arrays
file_names = np.array(file_names)
labels = np.array(labels)

# Save the labels
np.save("labels.npy", labels)
np.save("file_names.npy", file_names)

print(f"Extracted {len(labels)} labels. Distribution:")
print(f"Term (0): {np.sum(labels == 0)}")
print(f"Preterm (1): {np.sum(labels == 1)}")
print("Labels saved as 'labels.npy'.")
