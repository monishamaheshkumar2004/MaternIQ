import wfdb
import os
import glob
import numpy as np

# Set the path to the folder containing the dataset
folder_path = r"C:\Users\Pradeepa\Desktop\ehg_preterm\term-preterm-ehg-database-1.0.1\tpehgdb"

# Get all .dat files in the folder
dat_files = glob.glob(os.path.join(folder_path, "*.dat"))

# Sampling rate and segment selection
fs = 20  # Hz
start_sample = 5 * 60 * fs  # 6000 (Start at 5 minutes)
end_sample = 25 * 60 * fs   # 30000 (End at 25 minutes)

# Dictionary to store preprocessed data
preprocessed_data = {}

for dat_file in dat_files:
    record_name = os.path.splitext(dat_file)[0]  # Remove extension

    try:
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

        # Store the preprocessed data
        preprocessed_data[record_name] = filtered_signals

        # Print progress
        print(f"Processed: {record_name}, Shape: {filtered_signals.shape}")

    except Exception as e:
        print(f"Error processing {record_name}: {e}")

# Save the preprocessed data
np.save("preprocessed_ehg.npy", preprocessed_data)
print(f"\nTotal processed files: {len(preprocessed_data)}")
print("Preprocessing complete. Data saved as 'preprocessed_ehg.npy'.")
