import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed train and test data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Define hyperparameter grid for Random Forest
param_grid = {
    "n_estimators": [50, 100, 200],  # Number of trees in the forest
    "max_depth": [None, 10, 20, 30],  # Maximum depth of the trees
    "min_samples_split": [2, 5, 10],  # Minimum number of samples to split
    "min_samples_leaf": [1, 2, 4]  # Minimum samples at a leaf node
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search (5-fold cross-validation)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("\n✅ Best Hyperparameters:", best_params)

# Train the optimized model
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train, y_train)

# Make predictions
y_pred = best_rf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print final results
print("\n Final Fine-Tuned Random Forest Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f} (Higher recall improves preterm detection)")
print(f"F1-score: {f1:.4f}")

# Save the fine-tuned model (optional)
import joblib
joblib.dump(best_rf, "random_forest_best.pkl")
print("\nFine-Tuned Model Saved as 'random_forest_best.pkl'.")
