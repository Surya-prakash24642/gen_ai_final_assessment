import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text from a given PDF file."""
    try:
        doc = fitz.open(pdf_path)
        texts = [page.get_text("text") for page in doc]
        return texts  
    except Exception as e:
        return [f"Error extracting text: {e}"]
