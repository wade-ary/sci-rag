# RAG PIPELINE FOR ACADEMIC PAPERS

This project is a Retrieval-Augmented Generation (RAG) pipeline designed to answer user queries based on academic papers. 
The process begins by cleaning the PDF, This cleaned text is then split into smaller chunks, which are each embedded using OpenAI’s text-embedding-ada-002 model. 
The resulting vector representations are normalized and stored in a FAISS index for efficient similarity search.

When a user submits a query, the query is embedded in the same space and compared to all stored chunk embeddings using cosine similarity. 
The top 10 most semantically relevant chunks are retrieved from the FAISS index. These chunks are then passed to LlamaIndex, 
which further reranks them using its own internal node representation and retrieval mechanisms to select the top 5 most contextually relevant pieces.

Finally, these top 5 chunks, along with the original query, are passed to GPT-4 to generate a concise, coherent answer grounded in the source material. 
This setup allows for efficient and accurate academic question answering, even from long and complex documents.

Ideal for research assistance, literature reviews, or question answering from long-form academic documents.

## Instructions to run:


