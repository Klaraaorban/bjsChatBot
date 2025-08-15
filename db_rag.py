import contextlib
import io
import logging
import os
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np
import re

from yaml import warnings

# ----------------------------
# Config
# ----------------------------
PATH = "data/11.010-1-1.de.pdf"
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "bjs_collection"
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"

# ----------------------------
# Load and split documents
# ----------------------------
def load_documents():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        loader = UnstructuredPDFLoader(PATH, language="de")
        documents = loader.load()
        return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

# ----------------------------
# Filtering & embeddings
# ----------------------------
def is_dense_text(text, min_word_ratio=0.7):
    words = text.split()
    if not words:
        return False
    num_alpha = sum(1 for w in words if re.search(r'[A-Za-zÄÖÜäöüß]', w))
    return (num_alpha / len(words)) > min_word_ratio

def embedding_filter(chunks, embeddings):
    filtered = [c for c in chunks if len(c.page_content.split()) > 20]
    dense_chunks = [c for c in filtered if is_dense_text(c.page_content)]
    chunk_embeddings = [embeddings.embed_query(c.page_content) for c in dense_chunks]
    return dense_chunks, chunk_embeddings

# ----------------------------
# Save to Chroma
# ----------------------------
def save_to_chroma(dense_chunks, chunk_embeddings):
    client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_DIR,
            allow_reset=False
        )
    )
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = [c.page_content for c in dense_chunks]
    metadatas = [c.metadata for c in dense_chunks]
    ids = [f"chunk_{i}" for i in range(len(dense_chunks))]

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=chunk_embeddings
    )
    print("All chunks added to Chroma!")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    logging.getLogger("langchain").setLevel(logging.ERROR)
    logging.getLogger("unstructured").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    docs = load_documents()
    chunks = split_text(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    dense_chunks, chunk_embeddings = embedding_filter(chunks, embeddings)

    save_to_chroma(dense_chunks, chunk_embeddings)
