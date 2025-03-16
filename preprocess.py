import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load dataset
df = pd.read_csv("accident.csv")  # Update with the correct file path

# Strip spaces from column names
df.columns = df.columns.str.strip()

# Print available columns for debugging
print("Available columns:", df.columns)

# Identify categorical columns that need encoding
categorical_cols = ["Weather", "Road_Type", "Time_of_Day", "Road_Condition", "Vehicle_Type", "Accident_Severity"]

# If 'Road_Light_Condition' exists, include it
if "Road_Light_Condition" in df.columns:
    categorical_cols.append("Road_Light_Condition")

# Label Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Convert categorical values to numeric
    label_encoders[col] = le

# Save encoders for Flask app
joblib.dump(label_encoders, "label_encoders.pkl")

# Select Features and Target
if "Accident" in df.columns:
    X = df.drop(columns=["Accident"])
    y = df["Accident"]
else:
    raise ValueError("❌ 'Accident' column is missing in the dataset!")

# Convert all columns to numeric to avoid conversion errors
X = X.apply(pd.to_numeric, errors='coerce')

# Fill any missing values that might appear after conversion
X.fillna(0, inplace=True)

# Scale numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for Flask app
joblib.dump(scaler, "scaler.pkl")

# Save processed data for model training
joblib.dump((X_scaled, y), "processed_data.pkl")

print("✅ Data Preprocessing Complete!")
