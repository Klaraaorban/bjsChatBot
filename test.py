import os
import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- Config ----------------
PATH = r"data/11.010-1-1.de.pdf"      # PDF path
CHROMA_DIR = r"chroma_db"             # Chroma DB folder
COLLECTION_NAME = "bjs_col"           # Collection name
EMBEDDING_MODEL = r"C:\bjsChatBot\multilingual-e5-base"  # HuggingFace model

# Ensure Chroma directory exists
os.makedirs(CHROMA_DIR, exist_ok=True)

# Initialize persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Check if collection exists, else create it
existing_collections = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME in existing_collections:
    print(f"Collection '{COLLECTION_NAME}' exists. Using cached data.")
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
else:
    print(f"Collection '{COLLECTION_NAME}' not found. Creating new collection...")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---------------- Load PDF ----------------
loader = PyPDFLoader(PATH)
documents = loader.load()
print(f"Loaded {len(documents)} documents from PDF.")

# ---------------- Split Documents ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
print(f"Split documents into {len(chunks)} chunks.")

# ---------------- Compute Embeddings ----------------
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
texts = [c.page_content for c in chunks]
chunk_embeddings = embeddings.embed_documents(texts)
print(f"Computed embeddings for {len(chunk_embeddings)} chunks.")

# ---------------- Prepare for Chroma ----------------
docs_to_add = [c.page_content for c in chunks]
metadata_to_add = [c.metadata or {} for c in chunks]
ids_to_add = [f"chunk_{i}" for i in range(len(chunks))]

# ---------------- Add to Chroma ----------------
collection.add(
    documents=docs_to_add,
    metadatas=metadata_to_add,
    ids=ids_to_add,
    embeddings=chunk_embeddings
)

print("All chunks added to Chroma successfully!")
