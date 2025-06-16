import numpy as np
import matplotlib.pyplot as plt

# Load the features file
file_path = r"C:\Users\Pradeepa\Desktop\ehg_preterm\features.npy"
data = np.load(file_path, allow_pickle=True)

# If data is 2D, plot the first feature column
if len(data.shape) > 1:
    plt.plot(data[:, 0])  # Plot first feature
    plt.title("Feature Visualization")
    plt.xlabel("Samples")
    plt.ylabel("Feature Value")
    plt.show()
else:
    print("Data is not 2D. Shape:", data.shape)
