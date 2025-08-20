from playwright.sync_api import sync_playwright
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
import time
import sys
import os
# sys.path.append(r"C:\bjsChatBot\crawling")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

from crawling.crawl_angular import extract_links, extract_content

# ---------------- Config ----------------
CHROMA_DIR = r"chroma_db"
COLLECTION_NAME = "bjs_colAngularBig"
EMBEDDING_MODEL = r"C:\bjsChatBot\models\multilingual-MiniLM"

# ---------------- Initialize persistent Chroma client ----------------
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
existing_collections = [c.name for c in chroma_client.list_collections()]
collection = (
    chroma_client.get_collection(name=COLLECTION_NAME)
    if COLLECTION_NAME in existing_collections
    else chroma_client.get_or_create_collection(name=COLLECTION_NAME)
)

# ---------------- Chunking & Embedding ----------------
def add_to_chroma(scraped_texts):
    # Chunk the scraped texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=200, length_function=len, add_start_index=True
    )

    chunks = []
    for url, text in scraped_texts.items():
        doc_chunks = text_splitter.split_text(text)
        # attach metadata with source URL
        chunks.extend([{"text": c, "metadata": {"source": url}} for c in doc_chunks])

    # Compute embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    chunk_embeddings = embeddings.embed_documents(texts)

    # Add to Chroma
    collection.add(
        documents=texts,
        metadatas=[c["metadata"] for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        embeddings=chunk_embeddings,
    )
    print(f"Added {len(chunks)} chunks to Chroma!")

# ---------------- Main ----------------
if __name__ == "__main__":
    baseurl = "https://refbejuso.tlex.ch/app/de/systematic/texts_of_law"
    all_links = extract_links(baseurl)
    all_content = extract_content(all_links)

    add_to_chroma(all_content)
