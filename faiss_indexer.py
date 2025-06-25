import openai
import faiss
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"))
    
def read_cleaned_text(txt_path):
    """Reads the cleaned text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def chunk_text(text, chunk_size=500):
    """Splits text into chunks of approximately `chunk_size` words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    embedding = response.data[0].embedding
    return np.array(embedding)


def normalize_vector(vec):
    """Normalizes a numpy vector to have unit length."""
    return vec / np.linalg.norm(vec)

def build_faiss_index(embeddings):
    """Builds a FAISS index from a list of embeddings."""
    dim = len(embeddings[0])  # dimension of embeddings
    index = faiss.IndexFlatIP(dim)  # Inner Product (for cosine sim)
    index.add(np.array(embeddings).astype('float32'))
    return index

def process_and_store(txt_path, index_save_path):
    """Full pipeline: Read text - chunk - embed - normalize - store in FAISS."""
    text = read_cleaned_text(txt_path)
    chunks = chunk_text(text)
    embeddings = []
    
    for chunk in chunks:
        embedding = get_embedding(chunk)
        embedding = normalize_vector(embedding)
        embeddings.append(embedding)
    
    index = build_faiss_index(embeddings)
    faiss.write_index(index, index_save_path)
    
    print(f"FAISS index saved to {index_save_path}")
    return chunks  

