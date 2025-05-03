import pandas as pd
import joblib

# Load trained model
model = joblib.load("resume_jd_match_model.pkl")

# Load the labeled dataset
df = pd.read_csv("resume_jd_features_labeled.csv")

# Predict probabilities (confidence that label = 1)
df["match_score"] = model.predict_proba(df[["skill_overlap", "education_match", "experience_match"]])[:, 1]

# Get top-N matches (e.g. top 10)
top_n = 5

top_matches = df.sort_values(
    by=["match_score"],
    ascending=[False]
).head(top_n)



# Show result
print("🔍 Top Job Recommendations:")
print(top_matches[["job_title", "match_score", "skill_overlap", "education_match", "experience_match"]])
