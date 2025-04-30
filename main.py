from faiss_indexer import read_cleaned_text, chunk_text
from query_retriever import get_query_embedding, search_faiss_index
from llama_re_ranker import rerank_chunks
from final_answer_generator import generate_summary

#Load and chunk
text = read_cleaned_text("cleaned_output.txt")
chunks = chunk_text(text)

#Retrieve top 10 from FAISS
query = "What are the patterns of train attacks in Europe?"
query_embedding = get_query_embedding(query)
top_10_indices = search_faiss_index("index.faiss", query_embedding, top_k=10)
faiss_chunks = [chunks[i] for i in top_10_indices]

#Rerank with LlamaIndex
reranked_chunks = rerank_chunks(faiss_chunks, query)

#Print Top 5 Reranked Chunks
print("\n🔍 Top 5 Reranked Chunks:\n")
for i, chunk in enumerate(reranked_chunks, 1):
    print(f"[{i}] {chunk[:400]}...\n")

#Final summary via GPT
print("Final GPT Summary:\n")
summary = generate_summary(reranked_chunks, query)
print(summary)



