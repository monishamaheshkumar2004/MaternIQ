#%%
print("Script started successfully!")

import sys
sys.path.append(r"C:\Users\Pradeepa\AppData\Local\Programs\Python\Python310\lib\site-packages")  # Ensure Python finds installed packages

import wfdb
import os
import glob

import numpy as np
from PyEMD import EMD  # import for Empirical Mode Decomposition
import scipy.stats

print("PyEMD module loaded successfully!")

# Set the path to the dataset folder
folder_path = r"C:\Users\Pradeepa\Desktop\ehg_preterm\term-preterm-ehg-database-1.0.1\tpehgdb"

# Get all .dat files in the folder
dat_files = glob.glob(os.path.join(folder_path, "*.dat"))

print("Checking dataset folder...")
print(f"Found {len(dat_files)} .dat files in folder: {folder_path}")

if len(dat_files) == 0:
    print("No .dat files found! Check if the folder path is correct.")
    exit()

print("Script is running correctly. Proceeding with feature extraction...")

# Sampling rate and segment selection
fs = 20  # Hz
start_sample = 5 * 60 * fs  # 6000 (Start at 5 minutes)
end_sample = 25 * 60 * fs   # 30000 (End at 25 minutes)

# Dictionary to store preprocessed data
preprocessed_data = {}

for dat_file in dat_files:
    record_name = os.path.splitext(dat_file)[0]  # Remove extension

    try:
        print(f"Processing: {record_name}")  # Debugging print

        # Read the WFDB record
        record = wfdb.rdrecord(record_name)

        # Extract signal names
        channels = record.sig_name

        # Select only the filtered channels (0.08 - 4 Hz)
        selected_channels = [i for i, name in enumerate(channels) if "DOCFILT-4-0.08-4" in name]
        
        if len(selected_channels) == 0:
            print(f"Skipping {record_name}, no filtered channels found.")
            continue

        # Extract signals from selected channels
        filtered_signals = record.p_signal[:, selected_channels]

        # Keep only the middle 20 minutes
        filtered_signals = filtered_signals[start_sample:end_sample, :]

        # Store preprocessed data
        preprocessed_data[record_name] = filtered_signals

        print(f"Processed: {record_name}, Shape: {filtered_signals.shape}")  # Debugging print

    except Exception as e:
        print(f"Error processing {record_name}: {e}")

# Save the preprocessed data
np.save("preprocessed_ehg.npy", preprocessed_data)
print("Preprocessing complete. Data saved as 'preprocessed_ehg.npy'.")
#%%

#%%
import numpy as np
# Load preprocessed data for feature extraction
preprocessed_data = np.load("preprocessed_ehg.npy", allow_pickle=True).item()

# Feature extraction functions
def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))

def compute_teager_kaiser_energy(signal):
    return np.mean(signal[1:-1]**2 - signal[:-2] * signal[2:])

def compute_wavelet_log_entropy(signal):
    return np.sum(np.log(signal**2 + 1e-10))  # Small value to avoid log(0)

def compute_shannon_entropy(signal):
    probability_distribution, _ = np.histogram(signal, bins=10, density=True)
    probability_distribution = probability_distribution[probability_distribution > 0]
    return -np.sum(probability_distribution * np.log2(probability_distribution))

def compute_katz_fractal_dimension(signal):
    L = np.sum(np.abs(np.diff(signal)))
    D = np.max(np.abs(signal - np.mean(signal)))
    return np.log10(L) / (np.log10(D + 1e-10) + np.log10(len(signal)))

def compute_hurst_exponent(signal):
    N = len(signal)
    mean_signal = np.cumsum(signal - np.mean(signal))
    R = np.max(mean_signal) - np.min(mean_signal)
    S = np.std(signal)
    return np.log(R/S) / np.log(N) if S != 0 else 0

# Feature extraction using EMD
from PyEMD.EMD import EMD  # Correct import

feature_data = {}

print("Starting Feature Extraction...")
for subject, signals in preprocessed_data.items():
    subject_features = []
    
    for ch in range(signals.shape[1]):  # Iterate through 3 channels
        signal = signals[:, ch]

        # Apply Empirical Mode Decomposition (EMD)
        emd_instance = EMD()
        IMFs = emd_instance.emd(signal)  

        # Take only the first 4 IMFs (as per the paper)
        for imf in IMFs[:4]:  # Extract features from each IMF
            rms = compute_rms(imf)
            tke = compute_teager_kaiser_energy(imf)
            wle = compute_wavelet_log_entropy(imf)
            shannon = compute_shannon_entropy(imf)
            kfd = compute_katz_fractal_dimension(imf)
            hurst = compute_hurst_exponent(imf)
            
            subject_features.extend([rms, tke, wle, shannon, kfd, hurst])

    feature_data[subject] = subject_features

    print(f"Extracted features for {subject}: {len(subject_features)} values")

# Convert to NumPy and Save
features_array = np.array(list(feature_data.values()))
np.save("extracted_features.npy", features_array)

print("Feature extraction complete. Data saved as 'extracted_features.npy'.")







#%%
