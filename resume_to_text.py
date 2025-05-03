#Extract the data-

import pymupdf # imports the pymupdf library

doc = pymupdf.open("manmeet_resume_april.pdf") # open a document
for page in doc: # iterate the document pages
  text = page.get_text() # get plain text encoded as UTF-8

# Clean basic junk characters before sending to GenAI
import re

def basic_clean(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
    text = re.sub(r'\s+', ' ', text)            # Replace multiple spaces/newlines with one space
    return text.strip()

# Main usage
# pdf_path = "karan_resume_april.pdf"
# raw_text = extract_text_from_pdf(pdf_path)

cleaned_text = basic_clean(text)

# This cleaned_text is what you will now send to the GenAI model in your prompt
# print(cleaned_text)

#Gen AI model

import google.generativeai as genai

genai.configure(api_key="AIzaSyAUajoDJz5cinB3WlmZW06GtfSmPu8BVxk")

generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 2048
}

model = genai.GenerativeModel("models/gemini-2.5-pro-exp-03-25", generation_config==generation_config)

prompt = f"""
You are a resume parser and skill standardizer. Extract structured data from the resume text below, and ensure that all skills are mapped to **standardized technical terms** used in job descriptions.
Extract only the skills listed under the “SKILLS” section.
Ignore tools mentioned in projects unless they also appear in the SKILLS section.

⚠️ VERY IMPORTANT:
- Convert related skills into common terms.
- Do NOT keep duplicate or redundant forms.
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


response = model.generate_content(prompt)
csv_out = response.text

print(csv_out)

# csv
import csv
from io import StringIO

# Convert to a format Python can read as CSV
csv_data = StringIO(csv_out)

# Parse and save to CSV
with open("cleaned_resume_output.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    # Convert string data to list of rows
    rows = list(csv.reader(csv_data))

    # Write all rows except first and last
    for row in rows[1:-1]:
        writer.writerow(row)

print("Resume data saved to cleaned_resume_output.csv")


