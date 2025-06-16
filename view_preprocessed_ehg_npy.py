import numpy as np

#data = np.load("C:/Users/Pradeepa/Desktop/ehg_preterm/preprocessed_ehg.npy", allow_pickle=True)
#print(data)


# Load extracted features
features = np.load("C:/Users/Pradeepa/Desktop/ehg_preterm/extracted_features.npy", allow_pickle=True)

# Check the shape of the dataset
print("Feature matrix shape:", features.shape)

# Check first 5 feature sets
print("First 5 subjects' features:\n", features[:5])
