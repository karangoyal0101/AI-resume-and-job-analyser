import pandas as pd

# Load cleaned resume (with headers now)
resumes = pd.read_csv("cleaned_resume_output.csv")
jds = pd.read_csv("structured_job_descriptions_fast.csv")

# ----- Matching Functions -----
def jaccard_similarity(str1, str2):
    set1 = set(str(str1).lower().split(','))
    set2 = set(str(str2).lower().split(','))
    set1 = set(map(str.strip, set1))
    set2 = set(map(str.strip, set2))
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def degree_match(res_edu, jd_degree):
    return int(jd_degree.lower() in str(res_edu).lower()) if jd_degree != "Not specified" else 1

def experience_match(res_exp, jd_exp):
    if jd_exp == "Not specified":
        return 1
    try:
        jd_years = int(jd_exp.split()[0])
        return int(str(jd_years) in str(res_exp))
    except:
        return 0

# ----- Matching Logic -----
records = []
for res in resumes.itertuples():
    for jd in jds.itertuples():
        skill_sim = jaccard_similarity(res.skills, jd.required_skills)
        edu_match = degree_match(res.education, jd.required_degree)
        exp_match = experience_match(res.experience, jd.minimum_experience)

        records.append({
            "resume_name": res.name,
            "job_title": jd.job_title,
            "required skills": jd.required_skills,
            "skill_overlap": skill_sim,
            "education_match": edu_match,
            "experience_match": exp_match,
            "label" : 1 if skill_sim > 0.15 and edu_match == 1 and exp_match == 1 else 0
 # auto label
        })

df = pd.DataFrame(records)
# Recalculate labels safely (after full DataFrame is created)

# Label good matches only if all are true
df["label"] = df.apply(
    lambda row: 1 if row["skill_overlap"] > 0.15 and row["education_match"] == 1 and row["experience_match"] == 1 else 0,
    axis=1
)

print(df["label"].value_counts())

df.to_csv("resume_jd_features_labeled.csv", index=False)
print(f"✅ Matching complete. Saved to resume_jd_features_labeled.csv ({len(df)} rows)")

df_filtered = df[df["label"] == 1]
sorted_df_filtered = df_filtered.sort_values(by='skill_overlap',ascending=False)
sorted_df_filtered.to_csv("resume_jd_matches_filtered.csv", index=False)

