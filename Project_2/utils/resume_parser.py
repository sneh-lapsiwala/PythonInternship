# utils/resume_parser.py

import PyPDF2
import docx2txt
import fitz  # PyMuPDF

def extract_text_from_resume(file):
    """
    Extracts raw text from a resume file.
    Supports .pdf, .docx, and .txt formats.
    """
    if file.name.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() or "" for page in pdf.pages])
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    else:
        text = file.read().decode("utf-8")
    return text

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF using PyMuPDF (fitz).
    """
    text = ""
    try:
        file_bytes = uploaded_file.read()  # read the content once
        doc = fitz.open(stream=file_bytes, filetype="pdf")  # ✅ Accepts file-like object
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        text = f"Error reading PDF: {e}"
    return text

import pdfplumber

def extract_tables_from_pdf(file_obj):
    """
    Extracts tables from PDF using pdfplumber.
    """
    tables = []
    try:
        with pdfplumber.open(file_obj) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append(table)
    except Exception as e:
        tables.append([f"Error extracting tables: {e}"])
    return tables
