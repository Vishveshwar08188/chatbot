import PyPDF2

def load_pdf_text(pdf_path):
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
