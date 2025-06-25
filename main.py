import argparse
from pdf_cleaner import process_pdf
from faiss_indexer import read_cleaned_text, chunk_text, process_and_store
from query_retriever import get_query_embedding, search_faiss_index
from llama_re_ranker import rerank_chunks
from final_answer_generator import generate_summary
from dotenv import load_dotenv
import os
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline")
    parser.add_argument('-path', type=str, help='Path to the PDF file to process')
    parser.add_argument('-query', type=str, help='Query to be answered', default="What are the patterns of train attacks in Europe?")
    args = parser.parse_args()

    cleaned_txt_path = "cleaned_output.txt"
    index_path = "index.faiss"
    if args.path:
        process_pdf(args.path, cleaned_txt_path)

    # Ensure FAISS index exists
    if not os.path.exists(index_path):
        print(f"Index file {index_path} not found. Creating new index...")
        process_and_store(cleaned_txt_path, index_path)

    # Load and chunk
    text = read_cleaned_text(cleaned_txt_path)
    chunks = chunk_text(text)

    # Retrieve top 10 from FAISS
    query = args.query
    query_embedding = get_query_embedding(query)
    top_10_indices = search_faiss_index(index_path, query_embedding, top_k=10)
    faiss_chunks = [chunks[i] for i in top_10_indices]

    # Rerank with LlamaIndex
    reranked_chunks = rerank_chunks(faiss_chunks, query)

    # Print Top 5 Reranked Chunks
    print("\n Top 5 Reranked Chunks:\n")
    for i, chunk in enumerate(reranked_chunks, 1):
        print(f"[{i}] {chunk[:400]}...\n")

    # Final summary
    print("Final GPT Summary:\n")
    summary = generate_summary(reranked_chunks, query)
    print(summary)

    # Save query and response 
    with open("response.txt", 'w', encoding='utf-8') as f:
        f.write(f"Query: {query}\n")
        f.write(f"{summary}\n")
    print(f"\nQuery and response saved to response.txt")

if __name__ == "__main__":
    main()



