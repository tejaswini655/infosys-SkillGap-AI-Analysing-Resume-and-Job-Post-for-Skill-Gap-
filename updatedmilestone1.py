import streamlit as st
import pdfplumber
import docx

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Resume & JD Parser",
    layout="wide"
)

# ================= TITLE =================
st.title("📄 Resume and Job Description Parser")
st.write("Upload Resume and Job Description to get preview, skills, and job recommendation")

# ================= FUNCTIONS =================
def read_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def read_docx(uploaded_file):
    document = docx.Document(uploaded_file)
    return "\n".join([p.text for p in document.paragraphs])


def read_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def parse_file(uploaded_file):
    if uploaded_file is None:
        return ""

    uploaded_file.seek(0)
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return read_docx(uploaded_file)
    elif name.endswith(".txt"):
        return read_txt(uploaded_file)
    else:
        return ""

# ================= UI =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Upload Resume")
    resume_file = st.file_uploader(
        "Resume (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"]
    )

with col2:
    st.subheader("📌 Upload Job Description")
    jd_file = st.file_uploader(
        "Job Description (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"]
    )

st.divider()

# ================= PARSING =================
resume_text = parse_file(resume_file)
jd_text = parse_file(jd_file)

# ================= PREVIEW =================
st.subheader("🧾 Parsed Document Preview")

p1, p2 = st.columns(2)

with p1:
    st.markdown("### Resume Preview")
    if resume_text:
        st.text_area("Resume Text", resume_text, height=300)
    else:
        st.info("No resume uploaded")

with p2:
    st.markdown("### Job Description Preview")
    if jd_text:
        st.text_area("JD Text", jd_text, height=300)
    else:
        st.info("No job description uploaded")

st.divider()

# ================= SKILLS =================
COMMON_SKILLS = [
    "python", "java", "sql", "machine learning", "data analysis",
    "deep learning", "excel", "power bi", "tableau",
    "html", "css", "javascript", "react",
    "streamlit", "flask", "django", "aws", "cloud"
]

def extract_skills(text):
    text = text.lower()
    return sorted([skill for skill in COMMON_SKILLS if skill in text])

resume_skills = extract_skills(resume_text)
jd_skills = extract_skills(jd_text)

matched_skills = list(set(resume_skills) & set(jd_skills))
missing_skills = list(set(jd_skills) - set(resume_skills))

# ================= JOB RECOMMENDATION =================
def recommend_job(skills):
    skills = set(skills)

    if {"python", "machine learning", "data analysis"}.issubset(skills):
        return "Machine Learning Engineer"
    elif {"python", "sql", "data analysis"}.issubset(skills):
        return "Data Analyst"
    elif {"html", "css", "javascript"}.issubset(skills):
        return "Web Developer"
    elif {"python", "streamlit"}.issubset(skills):
        return "Python Developer"
    else:
        return "Entry Level Software Engineer"

# ================= RESULTS =================
st.subheader("🧠 Skill Analysis & Job Recommendation")

if resume_text and jd_text:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### ✅ Resume Skills")
        st.write(resume_skills if resume_skills else "No skills found")

    with c2:
        st.markdown("### 🎯 JD Skills")
        st.write(jd_skills if jd_skills else "No skills found")

    with c3:
        st.markdown("### 🔗 Matched Skills")
        st.write(matched_skills if matched_skills else "No match")

    st.markdown("### ❌ Missing Skills")
    st.write(missing_skills if missing_skills else "None 🎉")

    job = recommend_job(resume_skills)
    st.markdown("### 💼 Recommended Job Role")
    st.success(job)

    match_percent = int((len(matched_skills) / max(len(jd_skills), 1)) * 100)
    st.markdown("### 📊 Resume–JD Match Percentage")
    st.progress(match_percent)
    st.write(f"Match: {match_percent}%")

else:
    st.info("Please upload both Resume and Job Description")