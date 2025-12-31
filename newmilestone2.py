import sys
import streamlit as st

st.write("Python path:", sys.executable)
import pdfplumber
import docx
import spacy
from transformers import pipeline
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Skill Extraction using NLP", layout="wide")

# ================= LOAD MODELS =================
nlp = spacy.load("en_core_web_sm")
bert_ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# ================= TITLE =================
st.title("📄 Resume & Job Description Skill Extraction (NLP)")
st.write("Milestone 1: Parsing | Milestone 2: Skill Extraction using spaCy & BERT")

# ================= FILE READERS =================
def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")

def parse_file(file):
    if file is None:
        return ""
    file.seek(0)
    name = file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    elif name.endswith(".docx"):
        return read_docx(file)
    elif name.endswith(".txt"):
        return read_txt(file)
    return ""

# ================= SKILL LISTS =================
TECH_SKILLS = [
    "python","java","sql","machine learning","deep learning","nlp",
    "tensorflow","pytorch","scikit-learn","data analysis","data visualization",
    "aws","azure","gcp","power bi","tableau","statistics"
]

SOFT_SKILLS = [
    "communication","leadership","teamwork","problem solving",
    "critical thinking","adaptability","time management"
]

# ================= SKILL EXTRACTION =================
def extract_skills_spacy(text):
    doc = nlp(text.lower())
    skills = set()
    for chunk in doc.noun_chunks:
        if chunk.text in TECH_SKILLS or chunk.text in SOFT_SKILLS:
            skills.add(chunk.text)
    return skills

def extract_skills_bert(text):
    entities = bert_ner(text)
    skills = set()
    for ent in entities:
        word = ent["word"].lower()
        if word in TECH_SKILLS or word in SOFT_SKILLS:
            skills.add(word)
    return skills

def classify_skills(skills):
    tech = [s for s in skills if s in TECH_SKILLS]
    soft = [s for s in skills if s in SOFT_SKILLS]
    return tech, soft

# ================= UI =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Upload Resume")
    resume_file = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])

with col2:
    st.subheader("📌 Upload Job Description")
    jd_file = st.file_uploader("Job Description (PDF/DOCX/TXT)", type=["pdf","docx","txt"])

st.divider()

# ================= MILESTONE 1 OUTPUT =================
resume_text = parse_file(resume_file)
jd_text = parse_file(jd_file)

st.subheader("🧾 Milestone 1: Parsed Text Output")

c1, c2 = st.columns(2)
with c1:
    st.text_area("Resume Text", resume_text, height=250)

with c2:
    st.text_area("Job Description Text", jd_text, height=250)

st.divider()

# ================= MILESTONE 2 OUTPUT (MODIFIED) =================
st.subheader("🧠 Milestone 2: Skill Extraction using NLP")

if resume_text or jd_text:
    combined_text = resume_text + " " + jd_text

    spacy_skills = extract_skills_spacy(combined_text)
    bert_skills = extract_skills_bert(combined_text)

    all_skills = set(spacy_skills).union(set(bert_skills))
    tech_skills_list, soft_skills_list = classify_skills(all_skills)

    # ====== Layout: Left for extracted skills, Right for pie chart ======
    colA, colB = st.columns([2, 1])

    with colA:
        st.markdown("### ✅ Extracted Skills")
        st.markdown("**Technical Skills**")
        if tech_skills_list:
            for s in tech_skills_list:
                st.success(s)  # green box
        else:
            st.info("No technical skills found")

        st.markdown("**Soft Skills**")
        if soft_skills_list:
            for s in soft_skills_list:
                st.info(s)  # blue box
        else:
            st.info("No soft skills found")

    with colB:
        st.markdown("### 📊 Skill Distribution")
        labels = ["Technical Skills", "Soft Skills"]
        values = [len(tech_skills_list), len(soft_skills_list)]
        if sum(values) > 0:
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#2ecc71","#3498db"])
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.info("No skills to display in chart")

    st.divider()

    st.markdown("### 🏷️ Skill Tag Visualization")
    if all_skills:
        tags = "  ".join([f"`{skill}`" for skill in all_skills])
        st.markdown(tags)
    else:
        st.info("No skills extracted")

else:
    st.warning("Please upload Resume or Job Description")