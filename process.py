import wfdb
import os
import glob
import matplotlib.pyplot as plt

# Path to your dataset folder
folder_path = "C:/Users/Pradeepa/Desktop/ehg_preterm/term-preterm-ehg-database-1.0.1/tpehgdb"
 # Change this to your dataset path

# Get all .dat files in the folder
dat_files = glob.glob(os.path.join(folder_path, "*.dat"))

# Process each file
for dat_file in dat_files:
    # Get the record name (without extension)
    record_name = os.path.splitext(dat_file)[0]

    try:
        # Read the WFDB record
        record = wfdb.rdrecord(record_name)

        # Extract signals and metadata
        signals = record.p_signal  # Multichannel EHG signals
        fs = record.fs  # Sampling frequency
        channels = record.sig_name  # Channel names

        # Plot the signals from all channels
        plt.figure(figsize=(12, 6))
        for i in range(signals.shape[1]):
            plt.plot(signals[:, i], label=f'Channel {channels[i]}')

        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.title(f"EHG Signals from {os.path.basename(record_name)}")
        plt.legend()
        plt.show()

        # Print basic info
        print(f"Processed: {record_name}")
        print(f"Sampling Frequency: {fs} Hz")
        print(f"Channels: {channels}\n")

    except Exception as e:
        print(f"Error processing {record_name}: {e}")
