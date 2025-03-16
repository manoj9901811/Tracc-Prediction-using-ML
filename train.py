import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load processed data
X_scaled, y = joblib.load("processed_data.pkl")

# Convert y to numeric and handle NaN values
y = pd.Series(y).apply(pd.to_numeric, errors='coerce')  # Ensure it's numeric
y.fillna(y.mode()[0], inplace=True)  # Replace NaN with the most common value

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

# Save the trained model
joblib.dump(model, "accident_model.pkl")
