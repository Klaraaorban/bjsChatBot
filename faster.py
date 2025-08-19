from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import time

# ---------------- FastAPI setup ----------------
app = FastAPI()
class Query(BaseModel):
    question: str

# ---------------- Config ----------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bjs_col"
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 3

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

# ---------------- API endpoints ----------------
@app.get("/")
def root():
    return {"message": "Server is running. POST to /ask to query."}

@app.post("/ask")
def ask_endpoint(query: Query):
    start_time = time.time()

    # 1. Embed query
    query_embedding = embeddings.embed_query(query.question)

    # 2. Retrieve top chunks from Chroma
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    top_chunks = [
        {"content": c, "metadata": m}
        for c, m in zip(results["documents"][0], results["metadatas"][0])
    ]

    # 3. Ask Ollama
    answer = ask_ollama(query.question, top_chunks)

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Request processed in {runtime:.2f} seconds")

    return {"answer": answer, "runtime_sec": runtime}
