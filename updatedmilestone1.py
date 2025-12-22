import streamlit as st
import pdfplumber
import docx

# ---------- PAGE CONFIG (must be first Streamlit command) ----------
st.set_page_config(
    page_title="Resume & JD Parser",
    layout="wide"
)

# ---------- TITLE ----------
st.title("📄 Resume and Job Description Parser")
st.write("Upload a Resume and a Job Description to see parsed document preview")

# ---------- FUNCTIONS ----------
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

    # reset pointer (IMPORTANT)
    uploaded_file.seek(0)

    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return read_docx(uploaded_file)
    elif name.endswith(".txt"):
        return read_txt(uploaded_file)
    else:
        return "Unsupported file format"

# ---------- UI ----------
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

# ---------- PARSING ----------
resume_text = parse_file(resume_file)
jd_text = parse_file(jd_file)

# ---------- PREVIEW ----------
st.subheader("🧾 Parsed Document Preview")

p1, p2 = st.columns(2)

with p1:
    st.markdown("### Resume Preview")
    if resume_text:
        st.text_area("Resume Text", resume_text, height=350)
    else:
        st.info("No resume uploaded")

with p2:
    st.markdown("### Job Description Preview")
    if jd_text:
        st.text_area("JD Text", jd_text, height=350)
    else:
        st.info("No job description uploaded")

st.divider()

# ---------- SUMMARY ----------
if resume_text and jd_text:
    st.subheader("📊 Summary")
    st.write("Resume words:", len(resume_text.split()))
    st.write("JD words:", len(jd_text.split()))