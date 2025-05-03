import streamlit as st
import pandas as pd
import joblib
import fitz
import re
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
from openai import OpenAI
import ast
import os


#Client for gen-AI. We are using Google's Gemma 3-27-b free version with Openrouter.ai 

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key = st.secrets["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"],
)

#Skill keywords will be used to find skills from jd
# ai generated keywords
skill_keywords = [
    # Programming Languages
    "python", "java", "c++", "c", "javascript", "ruby", "go", "rust", "swift", "kotlin", 
    "typescript", "php", "shell scripting", "perl", "r", "matlab", "haskell", "objective-c", 
    "scala", "lua", "groovy", "elixir", "dart", "vhdl", "sql", "graphql", "json", "html", 
    "css", "xml", "yaml", "html5", "actionscript", "batch scripting", "xquery", "tcl",

    # Web Development Frameworks
    "django", "flask", "react", "angular", "vue.js", "node.js", "express.js", "ember.js", 
    "svelte", "next.js", "gatsby", "nuxt.js", "spring", "laravel", "rails", "zend", "backbone.js", 
    "bootstrap", "materialize", "tailwind css", "foundation", "apache", "nginx", "redis",

    # Front-End Development
    "html", "css", "javascript", "sass", "webpack", "babel", "typescript", "ajax", "jquery", 
    "responsive design", "cross-browser compatibility", "ui/ux design", "scss", "less", "react hooks", 
    "react native", "vuex", "redux", "material ui", "ionic", "swiftui", "flutter", "ember.js", 
    "angularjs", "css grid", "flexbox", "bootstrap", "tailwind", "javascript animations", "css animations",

    # Back-End Development
    "java", "python", "node.js", "c#", "go", "ruby", "php", "scala", "express.js", "spring", 
    "hibernate", "restful services", "microservices", "api development", "soap", "websockets", 
    "oauth2", "jwt", "grpc", "graphql", "mongodb", "postgresql", "mysql", "sqlite", "redis", 
    "couchdb", "firebase", "firebase cloud functions", "neo4j", "rabbitmq", "kafka", "message queuing",
    "apache kafka", "apache beam", "gcp", "aws lambda", "azure functions", "docker", "docker-compose", 
    "jenkins", "gitlab", "ci/cd", "jenkins pipelines", "cloudformation", "serverless architecture",

    # Data Engineering & Big Data
    "spark", "hadoop", "mapreduce", "kafka", "elasticsearch", "apache flink", "apache beam", 
    "storm", "kinesis", "bigquery", "etl", "etl pipelines", "data lakes", "data warehouses", 
    "big data analytics", "presto", "impala", "hive", "google cloud dataflow", "airflow", "dagster", 
    "data modeling", "aws s3", "hdfs", "parquet", "avro", "delta lake", "kubernetes", "containerization", 
    "cloud storage", "cloud computing", "docker", "pivotal cloud foundry", "databricks", "hadoop ecosystem", 
    "oracle", "cassandra", "couchbase", "sql server",

    # Machine Learning & AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn", 
    "xgboost", "lightgbm", "catboost", "reinforcement learning", "neural networks", "cnn", 
    "rnn", "lstm", "autoencoders", "gan", "svm", "decision trees", "random forest", "feature engineering", 
    "feature selection", "hyperparameter tuning", "cross-validation", "ensemble learning", "clustering", 
    "dimensionality reduction", "pca", "t-sne", "grid search", "bayesian optimization", "time series forecasting", 
    "computer vision", "opencv", "image processing", "nlp", "text classification", "sentiment analysis", 
    "chatbots", "speech recognition", "natural language generation", "word embeddings", "bert", "gpt", 
    "bert-based models", "transfer learning", "meta learning", "reinforcement learning", "model deployment", 
    "mlops", "ai ethics", "knowledge graphs", "semantic web", "data preprocessing", "data augmentation",

    # Cloud Technologies & DevOps
    "aws", "azure", "google cloud", "devops", "docker", "kubernetes", "terraform", "ansible", "chef", 
    "puppet", "ci/cd", "jenkins", "git", "gitlab", "github", "bitbucket", "cloudformation", "aws lambda", 
    "elastic beanstalk", "heroku", "monitoring", "prometheus", "grafana", "splunk", "new relic", 
    "datadog", "kibana", "logstash", "cloud security", "firewall", "cybersecurity", "vpn", "iso 27001", 
    "cloud native", "microservices architecture", "serverless architecture",

    # Databases & NoSQL
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "dynamodb", "neo4j", "elasticsearch", 
    "sqlite", "firebase", "mariadb", "cockroachdb", "graphql", "sql server", "oracle", "bigtable", 
    "couchdb", "hbase", "amazon rds", "snowflake", "google cloud sql", "cloud firestore", "noSQL", "sqlite3", 
    "sqlalchemy", "dbeaver", "dbt", "indexeddb", "riak", "redis clusters",

    # Testing & Quality Assurance
    "unit testing", "integration testing", "test-driven development", "pytest", "mocha", "junit", 
    "selenium", "cypress", "karma", "chai", "enzyme", "jest", "supertest", "load testing", "performance testing", 
    "mocking", "ci testing", "continuous testing", "security testing", "functional testing", "regression testing", 
    "manual testing", "selenium grid", "mocking frameworks", "code coverage", "test automation",

    # Security & Cryptography
    "cybersecurity", "encryption", "ssl/tls", "public key infrastructure", "firewall", "vpn", "iso 27001", 
    "ethical hacking", "penetration testing", "network security", "application security", "oauth2", "jwt", 
    "rsa encryption", "aes encryption", "hashing algorithms", "security protocols", "xss", "sql injection", 
    "ddos protection", "sso", "incident response", "penetration testing", "bug bounty", "data protection", 
    "identity management", "siem", "compliance", "gdpr",

    # Blockchain & Cryptocurrency
    "blockchain", "ethereum", "smart contracts", "solidity", "cryptocurrency", "bitcoin", "ethereum", 
    "nft", "decentralized applications", "web3", "ipfs", "crypto trading", "cryptography", "defi", "dao", 
    "hyperledger", "ripple", "litecoin", "solana", "polkadot", "cosmos", "blockchain development", 
    "proof of stake", "proof of work", "consensus algorithms", "private blockchain",

    # Artificial Intelligence & Robotics
    "robotics", "robot operating system (ros)", "robot control", "path planning", "reinforcement learning", 
    "robotic process automation", "computer vision", "artificial intelligence", "autonomous systems", 
    "speech processing", "natural language processing", "machine perception", "deep reinforcement learning",

    # Other Specialized Tools & Topics
    "vfx", "3d modeling", "blender", "maya", "3ds max", "unreal engine", "unity", "augmented reality", 
    "virtual reality", "iot", "edge computing", "quantum computing", "algorithms", "data structures", 
    "parallel computing", "mapreduce", "cloud architecture", "event-driven architecture", "data visualization", 
    "tableau", "power bi", "matplotlib", "plotly", "seaborn", "ggplot2", "d3.js", "plotly.js", "kibana",
    "deep reinforcement learning", "human-computer interaction", "ar/vr development", "edge AI", "ML pipelines"
]


# Load trained model
model = joblib.load("resume_jd_match_model.pkl")

# streamlit title and text.
st.title("🤖 AI Resume Matcher")
st.write("Upload your resume and job description list to find best job matches!")

# --- Upload Resume ---
resume_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# --- Upload Job Descriptions (Optional) ---
jd_file = st.file_uploader("Upload Job Descriptions CSV (optional)", type=["csv"])

# Load JD data with fallback
if jd_file is not None:
    st.success("✅ Using uploaded job description file")
    try:
        df = pd.read_csv(jd_file)
    except Exception as e:
        st.error(f"❌ Failed to read uploaded JD file: {e}")
        st.stop()
else:
    st.info("ℹ️ No JD uploaded. Using default JD data of linkedIN 2024")
    try:
        df = pd.read_csv("job_title_des.csv")
    except FileNotFoundError:
        st.error("❌ Default data not found in project folder.")
        st.stop()

# number of top matches
N = st.number_input("Enter number of top matches you want to see:", min_value=1, max_value=100, value=10, step=1, format="%d")

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file):
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def basic_clean(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces/newlines with one space
    return text.strip()


def extract_skills_from_jd(text, skill_list):
    text = text.lower()
    return [skill for skill in skill_list if skill in text]


def extract_experience(text):
    match = re.search(r'(\d+)\s+year', text.lower())
    return f"{match.group(1)} years" if match else "Not specified"

def extract_degree(text):
    text = text.lower()
    if "bachelor" in text:
        return "Bachelor's"
    elif "master" in text:
        return "Master's"
    elif "phd" in text:
        return "PhD"
    else:
        return "Not specified"

def jaccard_similarity(resume_skills, jd_skills):
    if not isinstance(resume_skills, str):
        resume_skills = ""
    if not isinstance(jd_skills, str):
        jd_skills = ""
    set1 = set(resume_skills.lower().split(','))
    set2 = set(jd_skills.lower().split(','))
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

def compute_features(resumes, jds):
    required_columns = ["name", "skills", "education", "experience"]
    for col in required_columns:
        if col not in resumes.columns:
            st.error(f"❌ Resume data is missing column: '{col}'")
            return pd.DataFrame()

    records = []

    for res in resumes.itertuples():
        if pd.isna(res.skills):
            continue  # Skip if skills are missing

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
                "label": 1 if skill_sim > 0.05 and edu_match == 1 and exp_match == 1 else 0
            })

    if not records:
        st.warning("⚠️ No matching records were created. Check resume or JD data.")
        return pd.DataFrame()

    return pd.DataFrame(records)


# --- MAIN BUTTON ---
if st.button("Find Matching Jobs"):
    if resume_file:
        # Extract and clean resume text
        raw_text = extract_text_from_pdf(resume_file)
        cleaned_text = basic_clean(raw_text)

        with st.expander("👀 See Extracted Resume Text"):
            st.write(cleaned_text)

        prompt = f"""
        You are a resume parser and skill standardizer. Extract structured data from the resume text below, and ensure that all skills are mapped to **standardized technical terms** used in job descriptions.
        Extract only the skills listed under the “SKILLS” section.
        Ignore tools mentioned in projects unless they also appear in the SKILLS section.

        ⚠️ VERY IMPORTANT:
        - Convert related skills into common terms.
        - Do NOT keep duplicate or redundant forms.
        - Return valid CSV with 6 fields exactly. Wrap skills, education, and experience in double quotes so commas inside them don't break the format.
        - strictly follow this format- ["name","email","Phone-no.","skill 1,skill 2, skill 3","experience 1, experience2"]
        - no "" inside skills and experiences.
        - Do NOT include creative software unless it's technical (e.g., replace "DaVinci Resolve" with "video editing").

        Skill normalization examples:
        - AI/ML, Artificial Intelligence → machine learning
        - Num-Py, NumPy → numpy
        - OOPS → object oriented programming
        - Game Development and Designing → game development
        - Power BI (DAX) → power bi
        - Davinci Resolve, Adobe Premiere Pro → video editing
        - Open Ai → openai
        - Web Browser, Automation Libraries → automation
        - IR sensors, ultrasonic sensors → sensors

        Return only a **single CSV row** with fields:
        name, email, phone, skills (comma-separated), education, experience

        Resume text:
        \"\"\"{cleaned_text}\"\"\"
        """

        try:
            completion = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",
                messages=[{"role": "user", "content": prompt}]
            )
            # Safely check if response exists and has content
            if completion and completion.choices and completion.choices[0].message:
                resume_data = completion.choices[0].message.content
            else:
                st.error("❌ No valid response from GenAI. Try again or check your prompt/API.")
                st.stop()

        except Exception as e:
            st.error(f"❌ API call failed: {e}")
            st.stop()    


        resume_data = resume_data.replace("```csv", "").replace("```", "").strip()

        csv_data = StringIO(resume_data)
        csv_data.seek(0)

        with st.expander("🧠 GenAI Response"):
            st.code(resume_data, language="text")



        with open("cleaned_resume_output.csv", "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Remove markdown/backtick/code block junk
            cleaned = resume_data.strip("`").strip()

            # Wrap in brackets to parse it as a Python list safely
            try:
                # Ensure it's a valid list-like string
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = "[" + cleaned + "]"

                # Convert the string to an actual list (safely parses embedded commas)
                parsed_row = ast.literal_eval(cleaned)

                if isinstance(parsed_row, list) and len(parsed_row) == 6:
                    writer.writerow(["name", "email", "phone", "skills", "education", "experience"])
                    writer.writerow(parsed_row)
                else:
                    st.error("❌ Resume data format invalid. Expected 6 fields in a single row.")
                    st.stop()

            except Exception as e:
                st.error(f"❌ Failed to parse GenAI response: {e}")
                st.stop()


        resumes = pd.read_csv("cleaned_resume_output.csv")

        # Load JD data

        structured = []
        for i, row in df.iterrows():
            jd_text = str(row["Job Description"]).lower()
            
            skills = extract_skills_from_jd(jd_text, skill_keywords)
            exp = extract_experience(jd_text)
            degree = extract_degree(jd_text)
            job_title = row.get("Job Title", "Unknown")

            structured.append({
                "job_title": job_title,
                "required_skills": ", ".join(skills),
                "minimum_experience": exp,
                "required_degree": degree
            })

        jds = pd.DataFrame(structured)

        
        # Compute features and predict
        features_df = compute_features(resumes, jds)

        if features_df.empty:
            st.error("❌ No feature data created. Ensure your resume contains valid skills and experience.")
            st.stop()


        required_cols = ["skill_overlap", "education_match", "experience_match"]
        if not all(col in features_df.columns for col in required_cols):
            st.error("❌ Feature extraction failed. Missing columns in computed data.")
            st.stop()

        features_df["match_score"] = model.predict_proba(
            features_df[["skill_overlap", "education_match", "experience_match"]]
        )[:, 1]

        


        # Compute match score as percentage
        features_df["match_score"] = (features_df["match_score"] * 100).round(2)
        features_df["match_score"] = features_df["match_score"].astype(str) + " %"

        # Now filter top matches using original (unformatted) match_score for logic
        # But use float for comparison, so we extract numeric part
        features_df["match_score_value"] = features_df["match_score"].str.replace(" %", "").astype(float)

        top_matches = features_df[features_df["match_score_value"] >= 10]  # 10 instead of 0.1 * 100
        top_matches = top_matches.sort_values(by=["match_score_value", "skill_overlap"], ascending=False).head(N)

        # Drop helper column before showing
        top_matches.drop(columns=["match_score_value"], inplace=True)

        if top_matches.empty:
            st.error("❌ No good job matches found. Try improving your resume or uploading more relevant job descriptions.")
        else:
            st.success("✅ Top job matches found!")
            st.dataframe(top_matches)

            # Download button
            csv = top_matches.to_csv(index=False).encode("utf-8")
            st.download_button("Download Matches as CSV", csv, "job_matches.csv", "text/csv")
