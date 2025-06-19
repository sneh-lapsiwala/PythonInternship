# Resume parser
import PyPDF2
import docx2txt

def extract_text_from_resume(file):
    if file.name.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        text = "".join([page.extract_text() for page in pdf.pages])
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    else:
        text = file.read().decode("utf-8")
    return text
