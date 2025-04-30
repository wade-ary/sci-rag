from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
    project=os.getenv("OPENAI_PROJECT_ID"),
    organization= os.getenv("OPENAI_ORG_ID"))

def generate_summary(reranked_chunks, query):
    """Summarizes answer using top reranked chunks and GPT-4."""
    context = "\n\n".join(reranked_chunks)

    prompt = f"""
You are an academic assistant. Answer the following query only using the text provided no outside resources.
The text given might be broken so you will have to paraphrase it make sure to only use the context in provided text.


Query: {query}

Context:
{context}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt.strip()}
        ]
    )

    return response.choices[0].message.content
