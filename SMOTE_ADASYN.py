import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

# Load fixed features and labels
X = np.load(r"C:\Users\Pradeepa\Desktop\ehg_preterm\features.npy")
y = np.load(r"C:\Users\Pradeepa\Desktop\ehg_preterm\labels.npy")

# Apply SMOTE to generate synthetic samples
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Apply ADASYN for further refinement
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_balanced, y_balanced = adasyn.fit_resample(X_smote, y_smote)

# Save the balanced dataset
np.save(r"C:\Users\Pradeepa\Desktop\ehg_preterm\features_balanced.npy", X_balanced)
np.save(r"C:\Users\Pradeepa\Desktop\ehg_preterm\labels_balanced.npy", y_balanced)

print("SMOTE + ADASYN applied: Balanced dataset saved.")

# Count and print new class distribution
term_count = np.sum(y_balanced == 0)
preterm_count = np.sum(y_balanced == 1)

print(f"Term Births (After Balancing): {term_count}")
print(f"Preterm Births (After Balancing): {preterm_count}")
