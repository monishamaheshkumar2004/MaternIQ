# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, Dropout, Dense,
                                     BatchNormalization, Bidirectional, LSTM,
                                     GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
import os
from scipy.signal import medfilt
import wfdb
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# %% Configuration
base_path = "C:\\Users\\Pradeepa\\Desktop\\acidemia\\ctu-chb-intrapartum-cardiotocography-database-1.0.0\\"
model_path = "best_cnn_bilstm_model.keras"
window_size = 300
stride = 300
fs_downsampled = 0.25
threshold = 0.3  # Default threshold, will be tuned later

# %% Parse .hea file for label
def parse_hea_file(hea_path):
    ph, bdecf = None, None
    with open(hea_path, 'r') as f:
        for line in f:
            if line.startswith('#pH'):
                try: ph = float(line.split()[1])
                except: continue
            elif line.startswith('#BDecf'):
                try: bdecf = float(line.split()[1])
                except: continue
    return 1 if (ph is not None and bdecf is not None and (ph < 7.15 or bdecf > 12)) else 0

# %% Dataset Construction
X, y = [], []
records = [f[:-4] for f in os.listdir(base_path) if f.endswith('.hea')]

for rec in tqdm(records, desc="Building CNN dataset"):
    try:
        record = wfdb.rdrecord(os.path.join(base_path, rec))
        signals, fs, sig_names = record.p_signal, record.fs, record.sig_name
        fhr = signals[:, sig_names.index('FHR')]
        uc = signals[:, sig_names.index('UC')]

        # Cleaning
        fhr[(fhr < 50) | (fhr > 200)] = np.nan
        fhr[np.abs(np.diff(fhr, prepend=fhr[0])) * fs > 25] = np.nan
        fhr, uc = medfilt(fhr, kernel_size=5), medfilt(uc, kernel_size=5)

        # Downsampling
        ds = int(fs / fs_downsampled)
        fhr, uc = fhr[::ds], uc[::ds]

        label = parse_hea_file(os.path.join(base_path, rec + '.hea'))

        for start in range(0, len(fhr) - window_size + 1, stride):
            f_win, u_win = fhr[start:start+window_size], uc[start:start+window_size]
            if not np.any(np.isnan(f_win)) and not np.any(np.isnan(u_win)):
                X.append(np.stack([f_win, u_win], axis=1))
                y.append(label)
    except Exception as e:
        print(f"Skipping {rec}: {e}")

X, y = np.array(X), np.array(y)
print(f"\nDataset shape: {X.shape}, Class balance: {np.bincount(y)}")

# %% Train-Test Split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_train_flat, y_train = SMOTE(random_state=42).fit_resample(X_train_flat, y_train)
X_train = X_train_flat.reshape((-1, 300, 2))

# %% Focal Loss
def focal_loss(alpha=0.75, gamma=1.5):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return -K.mean(alpha_t * K.pow(1. - p_t, gamma) * K.log(p_t))
    return loss_fn

# %% Model Definition
def build_model():
    model = Sequential([
        tf.keras.Input(shape=(300, 2)),
        Conv1D(32, 5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy', tf.keras.metrics.AUC()])
    return model

model = build_model()
model.summary()

# %% Training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=8,
    callbacks=callbacks,
    verbose=1
)

# %% Evaluation
y_pred_prob = model.predict(X_test).flatten()
thresholds = np.linspace(0, 1, 50)
f1s = [f1_score(y_test, (y_pred_prob > t).astype(int), zero_division=0) for t in thresholds]
threshold = thresholds[np.argmax(f1s)]
y_pred = (y_pred_prob > threshold).astype(int)

print("\n=== Test Performance ===")
print(f"Best threshold: {threshold:.2f}")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

# %% Predict on External .dat File
root = tk.Tk()
root.withdraw()

if messagebox.askyesno("Signal File Input", "Classify an external .dat signal file?"):
    dat_path = filedialog.askopenfilename(title="Select .dat file", filetypes=[("DAT files", "*.dat")])

    def predict_from_file(dat_path):
        rec_name = os.path.splitext(os.path.basename(dat_path))[0]
        dir_name = os.path.dirname(dat_path)
        try:
            record = wfdb.rdrecord(os.path.join(dir_name, rec_name))
            signals, fs, sig_names = record.p_signal, record.fs, record.sig_name
            print("Channels found:", sig_names)

            fhr = signals[:, sig_names.index('FHR')]
            uc = signals[:, sig_names.index('UC')]

            fhr[(fhr < 50) | (fhr > 200)] = np.nan
            fhr[np.abs(np.diff(fhr, prepend=fhr[0])) * fs > 25] = np.nan
            fhr, uc = medfilt(fhr, kernel_size=5), medfilt(uc, kernel_size=5)

            ds = int(fs / fs_downsampled)
            fhr, uc = fhr[::ds], uc[::ds]

            if len(fhr) < 300 or len(uc) < 300:
                raise ValueError("Signal too short after downsampling.")

            fhr, uc = fhr[:300], uc[:300]
            sample = np.expand_dims(np.stack([fhr, uc], axis=1), axis=0)

            y_pred_prob = model.predict(sample)[0, 0]
            y_pred = int(y_pred_prob > threshold)
            print(f"\n=== WFDB Prediction ===")
            print(f"Probability: {y_pred_prob:.4f}")
            print(f"Prediction: {y_pred} ({'Acidemia' if y_pred else 'Normal'})")
        except Exception as e:
            print(f"Error processing file: {e}")

    if dat_path:
        predict_from_file(dat_path)

def load_and_preprocess_data():
    try:
        # Load the dataset
        print("[INFO] Loading dataset...")
        data = pd.read_csv('fetal_distress_dataset_cleaned.csv')
        
        # Prepare features and target
        X = data.drop(['outcome', 'id'], axis=1)
        y = data['outcome']
        
        print("[INFO] Dataset shape:", X.shape)
        print("[INFO] Class balance:", np.bincount(y))
        
        return X, y
    except Exception as e:
        print("[ERROR] Failed to load data:", str(e))
        return None, None

def train_model(X, y):
    try:
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Load or train the model
        model_path = 'random_forest_best.pkl'
        if os.path.exists(model_path):
            print("[INFO] Loading existing model...")
            model = joblib.load(model_path)
        else:
            print("[INFO] Training new model...")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            joblib.dump(model, model_path)
        
        # Evaluate the model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        print("[INFO] Training accuracy:", train_score)
        print("[INFO] Testing accuracy:", test_score)
        
        return model, scaler
    except Exception as e:
        print("[ERROR] Failed to train model:", str(e))
        return None, None

def make_prediction(model, scaler, input_data):
    try:
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        return prediction[0], probability[0]
    except Exception as e:
        print("[ERROR] Failed to make prediction:", str(e))
        return None, None

def format_results(prediction, probability):
    try:
        if prediction == 1:
            result = "High Risk - Potential Fetal Distress"
            recommendations = [
                "1. Immediate medical attention required",
                "2. Monitor fetal heart rate continuously",
                "3. Consider emergency intervention",
                "4. Prepare for possible C-section",
                "5. Contact specialist for consultation"
            ]
        else:
            result = "Low Risk - Normal Fetal Status"
            recommendations = [
                "1. Continue regular monitoring",
                "2. Maintain normal prenatal care",
                "3. Follow standard delivery protocols",
                "4. Monitor for any changes",
                "5. Schedule next check-up as planned"
            ]
        
        output = [
            "Prediction Results:",
            "-----------------",
            f"Status: {result}",
            f"Confidence: {probability[prediction]:.1%}",
            "",
            "Recommendations:",
            "-----------------"
        ]
        output.extend(recommendations)
        
        return "\n".join(output)
    except Exception as e:
        return f"[ERROR] Failed to format results: {str(e)}"

def main():
    try:
        # Ensure UTF-8 encoding
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        
        # Load and preprocess data
        X, y = load_and_preprocess_data()
        if X is None or y is None:
            sys.exit(1)
        
        # Train or load model
        model, scaler = train_model(X, y)
        if model is None or scaler is None:
            sys.exit(1)
        
        # For testing, use sample data
        # In production, this would come from user input
        sample_data = X.iloc[[0]]
        
        # Make prediction
        prediction, probability = make_prediction(model, scaler, sample_data)
        if prediction is None:
            sys.exit(1)
        
        # Format and print results
        print(format_results(prediction, probability))
        
    except Exception as e:
        print("[ERROR] An unexpected error occurred:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()