import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load training data
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# Train the Random Forest model again (if needed)
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "random_forest.pkl")

print("Random Forest model saved as 'random_forest.pkl'")
