import numpy as np
import faiss
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"))

def get_query_embedding(query, model="text-embedding-ada-002"):
    """Embed the query using OpenAI."""
    response = client.embeddings.create(
        input=[query],
        model=model
    )
    embedding = np.array(response.data[0].embedding)
    """Return normalized."""
    return embedding / np.linalg.norm(embedding) 

def search_faiss_index(index_path, query_embedding, top_k=10):
    index = faiss.read_index(index_path)
    query_embedding = query_embedding.astype('float32').reshape(1, -1)
    D, I = index.search(query_embedding, top_k)
    """Return top_k."""
    return I[0][:top_k]  

