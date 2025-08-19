# rag_ollama.py
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import faiss
import numpy as np
import re
import requests
import warnings
import logging
import io
import contextlib

# ----------------------------
# Config
# ----------------------------
PATH = "data/11.010-1-1.de.pdf"
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"
OLLAMA_URL = "http://localhost:11434/api/generate"
FAISS_TOP_K = 50  # number of nearest neighbors
OLLAMA_TOP_CHUNKS = 5  # number of top chunks to send to Ollama

# ----------------------------
# Utility functions
# ----------------------------
def load_documents(path=PATH):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        loader = UnstructuredPDFLoader(path, language="de")
        docs = loader.load()
    return docs

def split_text(documents, chunk_size=300, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def is_dense_text(text, min_word_ratio=0.7):
    words = text.split()
    if not words:
        return False
    num_alpha = sum(1 for w in words if re.search(r'[A-Za-zÄÖÜäöüß]', w))
    return (num_alpha / len(words)) > min_word_ratio

def embedding_filter(chunks, embeddings):
    """Filter chunks and compute embeddings."""
    # Filter long enough chunks
    filtered = [c for c in chunks if len(c.page_content.split()) > 20]
    # Keep dense text
    dense_chunks = [c for c in filtered if is_dense_text(c.page_content)]
    # Compute embeddings
    chunk_embeddings = [embeddings.embed_query(c.page_content) for c in dense_chunks]
    return dense_chunks, chunk_embeddings

def faiss_index(chunk_embeddings):
    matrix = np.array(chunk_embeddings).astype("float32")
    faiss.normalize_L2(matrix)
    d = matrix.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(matrix)
    faiss.write_index(index, "chunk_index.faiss")
    return index

def search_faiss(index, query_embedding, k=FAISS_TOP_K):
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

def ask_ollama(query: str, retrieved_chunks: list, model="llama3.2:3b") -> str:
    """Send a query to Ollama with context."""
    context = "\n\n".join([f"{c['content']}\n(Source: {c['metadata']['source']})"
                           for c in retrieved_chunks])
    prompt = f"""
Beantworte die Frage basierend auf dem folgenden Kontext. 
Wenn die Antwort nicht im Kontext steht, sage 'Ich weiß es nicht'.

Kontext:
{context}

Frage: {query}
Antwort:
"""
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

# ----------------------------
# Main pipeline
# ----------------------------
def main(query: str):

    warnings.filterwarnings("ignore")
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("unstructured").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    # 1. Load & split
    docs = load_documents()
    chunks = split_text(docs)

    # 2. Initialize embeddings & embed chunks
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    dense_chunks, chunk_embeddings = embedding_filter(chunks, embeddings)

    # 3. Build FAISS index
    index = faiss_index(chunk_embeddings)

    # 4. Embed query & search
    query_embedding = np.array([embeddings.embed_query(query)]).astype("float32")
    distances, indices = search_faiss(index, query_embedding, k=FAISS_TOP_K)
    ind_list = indices[0].tolist()

    # 5. Prepare top chunks for Ollama
    top_chunks = [
        {"content": dense_chunks[idx].page_content, "metadata": dense_chunks[idx].metadata}
        for idx in ind_list[:OLLAMA_TOP_CHUNKS]
    ]

    # 6. Ask Ollama
    answer = ask_ollama(query, top_chunks)
    print("Ollama says:", answer)

    # 7. Optional: print FAISS results
    # for idx, dist in zip(ind_list, distances[0].tolist()):
    #     chunk = dense_chunks[idx]
    #     print(f"Index: {idx}, Distance: {dist}")
    #     print(chunk.page_content)
    #     print("Metadata:", chunk.metadata)
    #     print("-" * 50)

if __name__ == "__main__":
    query = input("Stelle BJS eine Frage dar: ")
    main(query)
