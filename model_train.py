import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# Load your labeled feature data
df = pd.read_csv("resume_jd_features_labeled.csv")

# Safety check: drop rows with missing values
df.dropna(subset=["skill_overlap", "education_match", "experience_match", "label"], inplace=True)

# Convert label to int (in case it's float)
df["label"] = df["label"].astype(int)

# Define features & label
features = ["skill_overlap", "education_match", "experience_match"]
X = df[features]
y = df["label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("📋 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "resume_jd_match_model.pkl")
print("💾 Model saved as resume_jd_match_model.pkl")
