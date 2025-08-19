# faster.py
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import httpx
import time

# ---------------- FastAPI setup ----------------
app = FastAPI()
class Query(BaseModel):
    question: str

# ---------------- Config ----------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bjs_col"
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"  # keep your current model
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
TOP_K = 2  # fewer chunks → smaller prompt → faster response

# ---------------- Load Chroma ----------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# ---------------- Initialize Embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ---------------- In-memory cache ----------------
query_cache = {}

# ---------------- Helper: Ask Ollama asynchronously ----------------
async def ask_ollama_async(query: str, chunks: list) -> str:
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
    async with httpx.AsyncClient() as client:
        response = await client.post(OLLAMA_URL, json=payload, timeout=None)
        response.raise_for_status()
        return response.json()["response"]

# ---------------- API endpoints ----------------
@app.get("/")
def root():
    return {"message": "Server is running. POST to /ask to query."}

@app.post("/ask")
async def ask_endpoint(query: Query):
    start_time = time.time()

    # 1. Check cache first
    if query.question in query_cache:
        answer = query_cache[query.question]
    else:
        # 2. Embed query
        query_embedding = embeddings.embed_query(query.question)

        # 3. Retrieve top chunks from Chroma
        results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
        top_chunks = [
            {"content": c, "metadata": m}
            for c, m in zip(results["documents"][0], results["metadatas"][0])
        ]

        # 4. Ask Ollama asynchronously
        answer = await ask_ollama_async(query.question, top_chunks)

        # 5. Cache the answer
        query_cache[query.question] = answer

    end_time = time.time()
    runtime = end_time - start_time
    print(f"Request processed in {runtime:.2f} seconds")

    return {"answer": answer, "runtime_sec": runtime}
