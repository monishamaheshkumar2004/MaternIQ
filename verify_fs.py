import numpy as np

# Define feature names for 72 extracted features (assuming 4 IMFs per channel)
feature_names = []
for i in range(4):  # For 4 IMFs
    feature_names.extend([
        f"RMS_IMF{i+1}",
        f"PeakFreq_IMF{i+1}",
        f"MedianFreq_IMF{i+1}",
        f"ZeroCrossings_IMF{i+1}",
        f"Lyapunov_IMF{i+1}",
        f"CorrDim_IMF{i+1}",
        f"SampleEntropy_IMF{i+1}"
    ])

# Extend to match 72 features (assuming 3 channels)
feature_names = feature_names * 3  # Duplicate for 3 channels

# Load selected feature indices
selected_indices = np.load("selected_feature_indices.npy")

# Ensure indices do not exceed the feature_names list size
max_index = len(feature_names)
selected_indices = [i for i in selected_indices if i < max_index]

# Print feature names corresponding to selected indices
selected_feature_names = [feature_names[i] for i in selected_indices]

print("Selected Features:")
for i, name in enumerate(selected_feature_names):
    print(f"{i+1}. {name}")
