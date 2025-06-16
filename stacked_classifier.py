import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('svm', SVC(kernel='linear', probability=True, random_state=42)),
    ('adaboost', AdaBoostClassifier(n_estimators=100, random_state=42))
]

# Define meta-classifier (Logistic Regression)
meta_classifier = LogisticRegression()

# Create Stacked Classifier
stacked_model = StackingClassifier(estimators=base_models, final_estimator=meta_classifier, cv=5)

# Train the stacked model
stacked_model.fit(X_train, y_train)

# Make predictions
y_pred = stacked_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("\n📌 Stacked Classifier Performance:")
print(f"Accuracy: {accuracy:.4f} ✅ (Expected Improvement)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f} (Higher recall improves preterm detection)")
print(f"F1-score: {f1:.4f}")

# Save the stacked model
import joblib
joblib.dump(stacked_model, "stacked_classifier.pkl")
print("\n✅ Stacked Model Saved as 'stacked_classifier.pkl'.")
