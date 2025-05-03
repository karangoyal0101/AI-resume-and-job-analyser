# 🤖 AI Resume Matcher

This is a Streamlit-based web application that automatically matches resumes to job descriptions (JDs) using Natural Language Processing (NLP) and Gen- AI. It extracts key features from resumes and JDs and calculates similarity scores to rank how well candidates fit the given roles.

## 🚀 Features

- 📄 Upload and parse resumes (PDF format)
- 📊 Upload job descriptions (CSV format)
- 📌 Extract relevant features (skills, experience, etc.)
- 🧠 Uses **Jaccard Similarity** for matching candidate skills with JD requirements
- 📈 Ranks and scores resumes based on job fit
- 🌐 Integrated with OpenRouter’s `google/gemma-3-27b-it:free` API for resume analysing

## 📂 Project Structure

├── app.py # Main Streamlit web app
├── model_train.py # ML model training script
├── recommend_jobs.py # Matching and recommendation logic
├── resume_to_text.py # Resume PDF to text extraction
├── resume_id_analyser.py # JD/resume analyser (NLP)
├── jd_parcer_new.py # JD parsing logic
├── resume_id_match_model.pkl # Saved ML model for scoring
├── job_title_des.xlsx # Job descriptions dataset
├── cleaned_resume_output.xlsx # Processed resume features
├── training data/ # Folder for training data
├── requirements.txt # Project dependencies
└── venv39/ # Local Python virtual environment

## 🧠 Technologies Used

- Python
- Streamlit
- OpenAI-compatible LLM (via OpenRouter - Gemma 3B)
- NLP (SpaCy, Jaccard Similarity)
- Pandas

## 🚀 Getting Started

  1. Clone the repository
      git clone https://github.com/yourusername/resume-job-matcher.git
      cd resume-job-matcher
  2. Set up virtual environment
      python -m venv venv
      source venv/bin/activate  # or venv\Scripts\activate on Windows
  3. Install dependencies
      pip install -r requirements.txt
  4. Use Your API Key of Google's Gemma 3-27-b (use open-router.ai for fre)-
      copy-paste it in Code. 
  5. Run the app
      streamlit run app.py

📌 Usage
  Upload Your or sample resume in PDF format.

  Upload a Job Descriptions CSV file with columns like:
    Title, Company, Description, Skills
  Click "Match Resumes".
  View ranked resumes based on similarity to each job role.

How It Works
Text Extraction: Parses PDF resumes using PyMuPDF and regex.

Feature Engineering: Extracts entities such as skills, experience, and education using gen-AI(google's gemma).

Jaccard Similarity: Measures skill overlap between resumes and JDs.

LLM-Powered Parsing: Optionally enhances JD parsing via google/gemma-3-27b-it.

🔧 Scope for Improvement
✅ Resume scoring module with weights for experience, education, etc.

✅ Add UI filters (e.g., min similarity score)

🤖 Integrate semantic matching using embeddings (e.g., BERT)

🗂 Add database support for job and resume storage

📬 Export results as CSV or PDF

🔐 Add login/authentication for secure access

🧑‍💻 Author
Made with ❤️ by Karan Goyal
