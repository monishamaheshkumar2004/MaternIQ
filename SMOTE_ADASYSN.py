import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN

# Load features and labels
X = np.load("features.npy")
y = np.load("labels.npy")

# Apply SMOTE to generate synthetic samples
smote = SMOTE(sampling_strategy=0.75, random_state=42)  # Balances the dataset to 75% minority class
X_smote, y_smote = smote.fit_resample(X, y)

# Apply ADASYN for further refinement
adasyn = ADASYN(sampling_strategy='minority', random_state=42)  # Focuses on minority class
X_balanced, y_balanced = adasyn.fit_resample(X_smote, y_smote)

# Save the balanced dataset
np.save("features_balanced.npy", X_balanced)
np.save("labels_balanced.npy", y_balanced)

print("SMOTE + ADASYN applied: Balanced dataset saved.")

term_count = np.sum(y_balanced == 0)
preterm_count = np.sum(y_balanced == 1)

print(f"Term Births: {term_count}")
print(f"Preterm Births: {preterm_count}")