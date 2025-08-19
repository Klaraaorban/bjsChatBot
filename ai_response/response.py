# response.py
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import requests
import time
from ai_response.faster import FastAPI
from pydantic import BaseModel

start_time = time.time()
app = FastAPI()

class Query(BaseModel):
    question: str

# ---------------- Config ----------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bjs_col"
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"  # your local Ollama model
TOP_K = 3  # top chunks to send to Ollama

# ---------------- Load Chroma ----------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# ---------------- Initialize Embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------- Helper: Ask Ollama ----------------
def ask_ollama(query: str, chunks: list) -> str:
    context = "\n\n".join([f"{c['content']}\n(Source: {c.get('metadata', {}).get('source','unknown')})"
                           for c in chunks])
    prompt = f"""
Beantworte die Frage basierend auf folgendem Kontext.
Wenn die Antwort nicht im Kontext steht, sage 'Ich weiß es nicht'.

Kontext:
{context}

Frage: {query}
Antwort:
"""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]



# --------------- API for server ----------------
@app.post("/ask")
def ask(query: Query):
    query_embedding = embeddings.embed_query(query.question)
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    top_chunks = [
        {"content": c, "metadata": m}
        for c, m in zip(results["documents"][0], results["metadatas"][0])
    ]
    answer = ask_ollama(query.question, top_chunks)
    return {"answer": answer}

# ---------------- Main pipeline ----------------
def main():
    query = input("Stelle BJS eine Frage: ")

    # 1. Embed query
    query_embedding = embeddings.embed_query(query)

    # 2. Retrieve top chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    # Flatten Chroma result
    top_chunks = [
        {"content": c, "metadata": m}
        for c, m in zip(results["documents"][0], results["metadatas"][0])
    ]

    # 3. Ask Ollama
    answer = ask_ollama(query, top_chunks)
    print("\nOllama response:\n", answer)

if __name__ == "__main__":
    main()
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")
