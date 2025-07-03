# UPDATED app.py with wow features
from flask import Flask, request, render_template
import os
import docx2txt
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# --- Text Extraction Functions ---
def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def clean_text(text):
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = text.lower()
    return text

def extract_skills(text, skill_set):
    return [skill for skill in skill_set if skill in text]

# --- Skill Set (could be dynamic in future) ---
skill_keywords = [
    'python', 'java', 'sql', 'machine learning', 'deep learning',
    'data science', 'pandas', 'numpy', 'nlp', 'tensorflow', 'pytorch',
    'aws', 'docker', 'kubernetes', 'git', 'linux', 'flask', 'django'
]

# --- Flask Routes ---
@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        job_desc_clean = clean_text(job_description)
        job_skills = extract_skills(job_desc_clean, skill_keywords)

        resumes = []
        resume_names = []
        results = []

        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            resume_text = extract_text(filename)
            resume_clean = clean_text(resume_text)

            # TF-IDF Similarity
            vectorizer = TfidfVectorizer().fit_transform([job_desc_clean, resume_clean])
            vectors = vectorizer.toarray()
            cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

            # Skill Matching
            resume_skills = extract_skills(resume_clean, skill_keywords)
            matched = list(set(resume_skills) & set(job_skills))
            missing = list(set(job_skills) - set(resume_skills))
            skill_match_score = len(matched) / len(job_skills) if job_skills else 0

            # Final ATS Score
            final_score = round((cosine_sim * 0.5 + skill_match_score * 0.5) * 100, 2)

            results.append({
                'name': resume_file.filename,
                'cosine': round(cosine_sim * 100, 2),
                'skill_score': round(skill_match_score * 100, 2),
                'final_score': final_score,
                'matched_skills': matched,
                'missing_skills': missing
            })

        results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        return render_template('matchresume.html', results=results, job_skills=job_skills)

    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
