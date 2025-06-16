import numpy as np

# Load saved labels and file names
labels = np.load("labels.npy")
file_names = np.load("file_names.npy")  # This contains the subject IDs

# Print the first 10 records
for i in range(10):
    print(f"Subject: {file_names[i]}, Label: {labels[i]}")
