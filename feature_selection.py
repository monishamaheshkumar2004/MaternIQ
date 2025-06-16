import numpy as np
import scipy.stats as stats

# Load balanced features and labels
X = np.load("features_balanced.npy")  # Feature matrix after balancing
y = np.load("labels_balanced.npy")  # Labels after balancing

# Ensure data consistency
assert X.shape[0] == y.shape[0], "Mismatch between features and labels!"

# Perform Mann–Whitney U test for each feature
selected_features = []
p_values = []

for i in range(X.shape[1]):  # Iterate over all features
    term_values = X[y == 0, i]  # Features from term births
    preterm_values = X[y == 1, i]  # Features from preterm births
    
    # Perform Mann-Whitney U Test
    u_statistic, p_value = stats.mannwhitneyu(term_values, preterm_values, alternative='two-sided')
    
    # Store p-values
    p_values.append(p_value)

    # Select features with p-value ≤ 0.05
    if p_value <= 0.05:
        selected_features.append(i)

# Convert to NumPy array
X_selected = X[:, selected_features]  # Keep only selected features

# Save the filtered feature set
np.save("selected_features.npy", X_selected)
np.save("selected_feature_indices.npy", np.array(selected_features))

print(f" Selected {len(selected_features)} features out of {X.shape[1]}.")
print(f"Selected features saved as 'selected_features.npy'.")
