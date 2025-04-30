from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import os
# Safe: limit to single embedding at a time
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"))

embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key= os.getenv("OPENAI_API_KEY"),
    project= os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"),
    embed_batch_size=1
)


def rerank_chunks(chunks, query, top_k=5):
    """Rerank chunks using LlamaIndex"""
   

    documents = [Document(text=chunk) for chunk in chunks]
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    print(f"Embedding {len(nodes)} nodes.")

    index = VectorStoreIndex(nodes, embed_model=embed_model)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    results = retriever.retrieve(query)

    top_chunks = [res.node.get_content() for res in results]
    return top_chunks


