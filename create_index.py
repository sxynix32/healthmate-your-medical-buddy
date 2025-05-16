from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os

# Replace with your PDF file path - use raw string or forward slashes
PDF_PATH = r"data\lexi-comps-drug-information-handbook-17th-edition.pdf"
FAISS_INDEX_PATH = "faiss_index"

def create_faiss_index():
    # Load PDF documents
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Load embeddings model (can customize here)
    embeddings = HuggingFaceEmbeddings()

    # Create FAISS vectorstore from documents + embeddings
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save FAISS index locally
    if not os.path.exists(FAISS_INDEX_PATH):
        os.mkdir(FAISS_INDEX_PATH)
    vectorstore.save_local(FAISS_INDEX_PATH)

    print(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    create_faiss_index()