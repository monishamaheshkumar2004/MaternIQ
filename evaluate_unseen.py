import numpy as np
import joblib
import wfdb
import os
import scipy.stats as stats
import tkinter as tk
from tkinter import filedialog

# Load the trained model
model = joblib.load("random_forest.pkl")  # Ensure model is trained with 33 features

# Set dataset path
dataset_path = "C:/Users/Pradeepa/Desktop/ehg_preterm/term-preterm-ehg-database-1.0.1/tpehgdb"

# Load test set files (unseen files)
X_test = np.load("X_test.npy")  # Only real test samples
file_list = sorted([f for f in os.listdir(dataset_path) if f.endswith(".dat")])
unseen_files = file_list[:len(X_test)]  # Select only unseen files

# Function to preprocess EHG signals
def preprocess_ehg(file_path):
    """ Reads the .dat file, extracts filtered signals, and returns a processed feature vector. """
    record_name = os.path.splitext(file_path)[0]  # Remove extension
    try:
        # Read the EHG signal
        record = wfdb.rdrecord(record_name)

        # Extract filtered channels (same as used during training)
        channels = record.sig_name
        selected_channels = [i for i, name in enumerate(channels) if "DOCFILT-4-0.08-4" in name]

        if len(selected_channels) == 0:
            raise ValueError("No valid filtered channels found in the signal.")

        # Extract selected channels and take the middle 20-minute segment
        fs = 20  # Sampling frequency (Hz)
        start_sample = 5 * 60 * fs  # 5 min mark
        end_sample = 25 * 60 * fs   # 25 min mark
        signals = record.p_signal[start_sample:end_sample, selected_channels]

        # Compute features (same as training)
        feature_vector = extract_features(signals)
        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to extract the 33 selected features
def extract_features(signals):
    """ Extracts the 33 statistical and nonlinear features used in training. """
    feature_vector = []
    
    for ch in range(signals.shape[1]):  # Iterate through 3 channels
        signal = signals[:, ch]

        # Compute statistical features
        rms = np.sqrt(np.mean(signal**2))
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        skewness = stats.skew(signal)
        kurtosis = stats.kurtosis(signal)
        entropy = stats.entropy(np.histogram(signal, bins=10)[0] + 1e-10)  # Avoid log(0)
        iqr = np.percentile(signal, 75) - np.percentile(signal, 25)
        median = np.median(signal)
        variance = np.var(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)

        # Add features to vector (11 per channel × 3 channels = 33 features)
        feature_vector.extend([rms, mean_val, std_val, skewness, kurtosis, entropy, iqr, median, variance, min_val, max_val])

    return np.array(feature_vector)  # Should output 33 features

def get_actual_label(hea_file):
    """ Reads the .hea file and extracts the actual gestation week. """
    try:
        with open(hea_file, 'r') as file:
            for line in file:
                if "Gestation" in line:
                    parts = line.split()  # Split by spaces
                    for i, word in enumerate(parts):
                        if word.lower() == "gestation" and i + 1 < len(parts):
                            try:
                                gestation_week = float(parts[i + 1])  # Extract the number after "Gestation"
                                return 0 if gestation_week >= 37 else 1  # 0 = TERM, 1 = PRETERM
                            except ValueError:
                                continue  # Skip if conversion fails
    except Exception as e:
        print(f"Error reading {hea_file}: {e}")
    
    return None  # Return None if gestation week is not found

# Loop through unseen files and evaluate predictions
correct = 0
total = 0

print("\nEvaluating Unseen Files...\n")
for unseen_file in unseen_files:
    file_path = os.path.join(dataset_path, unseen_file)
    hea_path = file_path.replace(".dat", ".hea")  # Find the corresponding .hea file

    if not os.path.exists(hea_path):
        print(f"Skipping {unseen_file}: Missing .hea file.")
        continue

    # Get actual label from .hea file
    actual_label = get_actual_label(hea_path)
    if actual_label is None:
        print(f"Skipping {unseen_file}: Could not extract gestation week.")
        continue

    # Preprocess and extract features from unseen file
    X_new = preprocess_ehg(file_path)
    if X_new is None:
        print(f"Skipping {unseen_file}: Could not process signal.")
        continue

    # Ensure correct shape
    X_new = X_new.reshape(1, -1)

    if X_new.shape[1] != 33:
        print(f"Error: Extracted {X_new.shape[1]} features, but model expects 33 features.")
        continue

    # Make prediction
    prediction = model.predict(X_new)
    predicted_label = int(prediction[0])  # Convert to integer (0 = TERM, 1 = PRETERM)

    # Confidence scores
    probs = model.predict_proba(X_new)
    term_confidence = probs[0][0]
    preterm_confidence = probs[0][1]

    # Compare prediction with actual label
    match = "✅" if predicted_label == actual_label else "❌"
    if predicted_label == actual_label:
        correct += 1
    total += 1

    print(f"{match} {unseen_file}: Predicted = {'TERM' if predicted_label == 0 else 'PRETERM'}, "
          f"Actual = {'TERM' if actual_label == 0 else 'PRETERM'} "
          f"(Confidence: Term {term_confidence:.2f}, Preterm {preterm_confidence:.2f})")

# Print accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\nFinal Accuracy on Unseen Files: {accuracy:.2f}% ({correct}/{total} correct)\n")
else:
    print("\n⚠ No valid unseen files processed.\n")
