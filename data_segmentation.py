import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load selected features and labels
X = np.load(r"C:\Users\Pradeepa\Desktop\ehg_preterm\selected_features.npy")
y = np.load(r"C:\Users\Pradeepa\Desktop\ehg_preterm\labels_balanced.npy")  # This includes synthetic preterm data

# Load original labels to identify real preterm samples
y_original = np.load("labels.npy")

# Identify real preterm samples
real_preterm_indices = np.where(y_original == 1)[0]  # Indices of real preterm cases
real_preterm_X = X[real_preterm_indices]
real_preterm_y = y[real_preterm_indices]

# Identify synthetic preterm samples (all preterm cases in y_balanced that are NOT in real preterm)
synthetic_preterm_indices = np.where((y == 1) & (~np.isin(np.arange(len(y)), real_preterm_indices)))[0]
synthetic_preterm_X = X[synthetic_preterm_indices]
synthetic_preterm_y = y[synthetic_preterm_indices]

# Identify term cases
term_indices = np.where(y == 0)[0]
term_X = X[term_indices]
term_y = y[term_indices]

# Split term cases and synthetic preterm cases into training and testing (70%-30%)
X_train_term, X_test_term, y_train_term, y_test_term = train_test_split(term_X, term_y, test_size=0.3, random_state=42, stratify=term_y)
X_train_preterm, X_test_preterm, y_train_preterm, y_test_preterm = train_test_split(synthetic_preterm_X, synthetic_preterm_y, test_size=0.3, random_state=42, stratify=synthetic_preterm_y)

# Combine training data (Term + Synthetic Preterm)
X_train = np.vstack((X_train_term, X_train_preterm))
y_train = np.hstack((y_train_term, y_train_preterm))

# Combine test data (Term + **Real Preterm**)
X_test = np.vstack((X_test_term, real_preterm_X))
y_test = np.hstack((y_test_term, real_preterm_y))

# Normalize using Min-Max Scaling (0-1)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the final datasets
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Data Normalization & Splitting Complete!")
print(f"Training Data: {X_train.shape[0]} samples, Testing Data: {X_test.shape[0]} samples")
print(f"Training Label Distribution: Term: {np.sum(y_train == 0)}, Preterm (Synthetic): {np.sum(y_train == 1)}")
print(f"Testing Label Distribution: Term: {np.sum(y_test == 0)}, Preterm (Real): {np.sum(y_test == 1)}")
xdcd