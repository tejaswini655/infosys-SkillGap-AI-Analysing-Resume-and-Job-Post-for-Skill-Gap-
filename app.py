import streamlit as st
import pdfplumber
import docx

# ---------- Page Configuration ----------
st.set_page_config(page_title="Document Parser", layout="wide")

st.title("Document Parsing & Preview System")

# ---------- Functions ----------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file):
    document = docx.Document(file)
    text = ""
    for para in document.paragraphs:
        text += para.text + "\n"
    return text


def extract_text_from_txt(file):
    return file.read().decode("utf-8")


# ---------- Layout ----------
left_col, right_col = st.columns([1, 2])

# ---------- LEFT SIDE : Upload ----------
with left_col:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "doc", "docx", "txt"]
    )

# ---------- RIGHT SIDE : Output ----------
with right_col:
    st.subheader("Parsed Document Preview")

    if uploaded_file is not None:
        file_name = uploaded_file.name

        if file_name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)

        elif file_name.endswith(".docx") or file_name.endswith(".doc"):
            text = extract_text_from_docx(uploaded_file)

        elif file_name.endswith(".txt"):
            text = extract_text_from_txt(uploaded_file)

        else:
            text = "Unsupported file format."

        # Clean text
        cleaned_text = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )

        st.text_area(
            label="",
            value=cleaned_text,
            height=500
        )

    else:
        st.info("Upload a document to see the parsed content here.")