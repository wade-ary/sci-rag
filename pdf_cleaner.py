import re
import fitz 
import os

def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def remove_citations(text):
    """Removes in-text citations like [1], [2,3], (Smith et al., 2020)."""
    # Remove [number] citations
    text = re.sub(r'\[\d+(,\s*\d+)*\]', '', text)
    
    # Remove (Author, Year) citations
    text = re.sub(r'\([A-Za-z]+,?\s*(et al\.)?,?\s*\d{4}\)', '', text)

    return text

def clean_text(text):
    """Lowercases and removes citations."""
    text = text.lower()
    text = remove_citations(text)
    return text

def save_text_to_file(text, output_path):
    """Saves cleaned text to a .txt file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdf(pdf_path, output_txt_path):
    """Full pipeline: PDF - cleaned text - .txt"""
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    save_text_to_file(cleaned_text, output_txt_path)
    print(f" Successfully cleaned and saved to {output_txt_path}")

