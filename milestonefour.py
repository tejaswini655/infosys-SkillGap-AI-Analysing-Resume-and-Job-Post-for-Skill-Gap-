
import sys
import streamlit as st

st.write("Python path:", sys.executable)

import pdfplumber
import docx
import spacy
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Skill Gap Analysis using NLP", layout="wide")

# ================= LOAD MODELS =================
nlp = spacy.load("en_core_web_sm")
bert_ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
bert_embed = SentenceTransformer("all-MiniLM-L6-v2")

# ================= TITLE =================
st.title("📄 Resume & Job Description Skill Gap Analysis")
st.write("Milestone 1: Parsing | Milestone 2: Skill Extraction | Milestone 3: Skill Gap Analysis")

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
    soft = [s for s in SOFT_SKILLS]
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

# ================= MILESTONE 2 OUTPUT =================
st.subheader("🧠 Milestone 2: Skill Extraction using NLP")

if resume_text or jd_text:

    resume_skills = extract_skills_bert(resume_text)
    jd_skills = extract_skills_bert(jd_text)

    res_tech, res_soft = classify_skills(resume_skills)
    jd_tech, jd_soft = classify_skills(jd_skills)

    colA, colB = st.columns([2,1])

    with colA:
        st.markdown("### Resume Skills")
        for s in res_tech:
            st.success(s)
        for s in res_soft:
            st.info(s)

        st.markdown("### Job Description Skills")
        for s in jd_tech:
            st.success(s)
        for s in jd_soft:
            st.info(s)

    with colB:
        labels = ["Technical", "Soft"]
        values = [len(res_tech), len(res_soft)]
        if sum(values) > 0:
            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#2ecc71","#3498db"])
            ax.axis("equal")
            st.pyplot(fig)

    st.divider()

    # ================= MILESTONE 3: Side by Side =================
    st.subheader("📊 Milestone 3: Skill Gap Analysis & Similarity Matching")

    resume_list = list(resume_skills)
    jd_list = list(jd_skills)

    if resume_list and jd_list:

        res_emb = bert_embed.encode(resume_list)
        jd_emb = bert_embed.encode(jd_list)

        sim_matrix = cosine_similarity(res_emb, jd_emb)

        matched, partial, missing = [], [], []

        for i, skill in enumerate(jd_list):
            score = np.max(sim_matrix[:, i])
            if score >= 0.8:
                matched.append(skill)
            elif score >= 0.5:
                partial.append(skill)
            else:
                missing.append(skill)

        overall = int((len(matched)/len(jd_list))*100)

        # ---- Side by side layout ----
        col1, col2, col3 = st.columns([1.5, 1, 2])

        # Column 1: Metrics
        with col1:
            st.metric("Overall Match", f"{overall}%")
            st.metric("Matched Skills", len(matched))
            st.metric("Partial Matches", len(partial))
            st.metric("Missing Skills", len(missing))

        # Column 2: Donut chart
        with col2:
            fig, ax = plt.subplots()
            ax.pie(
                [len(matched), len(partial), len(missing)],
                labels=["Matched", "Partial", "Missing"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width":0.4},
                colors=["#2ecc71","#f1c40f","#e74c3c"]
            )
            ax.axis("equal")
            st.pyplot(fig)

        # Column 3: Similarity matrix + missing skills
        with col3:
            st.markdown("### 🔷 Similarity Matrix")
            df = pd.DataFrame(sim_matrix, index=resume_list, columns=jd_list)
            st.dataframe(df.round(2))

            st.markdown("### ⚠️ Missing Skills")
            if missing:
                for s in missing:
                    st.error(f"{s} | Priority: High")
            else:
                st.success("No missing skills")

    else:
        st.warning("Not enough skills for similarity analysis")

else:
    st.warning("Please upload Resume or Job Description")


# ================= MILESTONE 4 CODE =================
st.subheader("📈 Milestone 4: Dashboard")

# Prepare lists and values from previous milestones
all_skills = list(set(resume_list + jd_list))
resume_vals = [1 if s in resume_list else 0 for s in all_skills]
jd_vals = [1 if s in jd_list else 0 for s in all_skills]

# Side-by-side layout: Radar | Recommendations | Bar chart
col1, col2, col3 = st.columns([4,2,4])

# ----- Column 1: Radar / Pentagon Chart + Role Metrics -----
with col1:
    st.markdown("### 🛡️ Skills Comparison (Radar)")
    if all_skills:
        num_vars = len(all_skills)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        resume_plot = resume_vals + resume_vals[:1]
        jd_plot = jd_vals + jd_vals[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax.plot(angles, resume_plot, 'o-', linewidth=2, label='Resume', color="#2ecc71")
        ax.fill(angles, resume_plot, alpha=0.25, color="#2ecc71")
        ax.plot(angles, jd_plot, 'o-', linewidth=2, label='Job Description', color="#3498db")
        ax.fill(angles, jd_plot, alpha=0.25, color="#3498db")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_skills, fontsize=8)
        ax.set_yticks([0,1])
        ax.set_yticklabels([])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
        st.pyplot(fig)

    st.markdown("### 👤 Role Metrics")
    st.metric("Total Resume Skills", len(resume_list))
    st.metric("Total JD Skills", len(jd_list))
    st.metric("Overall Match", f"{overall}%")

# ----- Column 2: Upskilling Recommendations -----
with col2:
    st.markdown("### ⚡ Upskilling Recommendations")
    if missing:
        for skill in missing:
            st.warning(f"Learn: {skill}")
    else:
        st.success("No upskilling required")

# ----- Column 3: Zigzag Bar Chart + Metrics + Refresh/Settings/Download -----
with col3:
    colR1, colR2, colR3 = st.columns([6,1,1])
    with colR2: st.button("🔄 Refresh")
    with colR3: st.button("⚙️ Settings")

    colM1, colM2, colM3 = st.columns(3)
    colM1.metric("Overall Match", f"{overall}%")
    colM2.metric("Matched Skills", len(matched))
    colM3.metric("Missing Skills", len(missing))

    fig, ax = plt.subplots(figsize=(5,4))
    idx = np.arange(len(all_skills))
    bar_width = 0.35
    # Zigzag effect
    resume_shift = [0.1 if i%2==0 else 0 for i in range(len(all_skills))]
    jd_shift = [0.1 if i%2!=0 else 0 for i in range(len(all_skills))]

    ax.bar(idx, [v+shift for v, shift in zip(resume_vals, resume_shift)],
           bar_width, label="Resume", color="#2ecc71")
    ax.bar(idx + bar_width, [v+shift for v, shift in zip(jd_vals, jd_shift)],
           bar_width, label="Job Description", color="#3498db")

    ax.set_xticks(idx + bar_width/2)
    ax.set_xticklabels(all_skills, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0,1.5)
    ax.set_ylabel("Skill Presence")
    ax.set_title("Resume vs JD Skills Comparison")
    ax.legend()
    st.pyplot(fig)

    # CSV download
    report_df = pd.DataFrame({
        "Skill": all_skills,
        "In Resume": resume_vals,
        "In JD": jd_vals
    })
    csv = report_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV Report", csv, "skill_gap_report.csv", "text/csv")