import streamlit as st
import os
import requests
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Google Drive direct download links for your files:
FAISS_INDEX_URL = "https://drive.google.com/uc?export=download&id=1w3lxcjluJm2Chtkzg8eKL4AL-n3-_bg7"
PKL_INDEX_URL = "https://drive.google.com/uc?export=download&id=1pkYHbRViooKvnFhL8BGGrQleT3BmRphG"

# Local paths to save the downloaded index files
INDEX_DIR = Path("faiss_index")
INDEX_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
PKL_INDEX_PATH = INDEX_DIR / "index.pkl"

def download_file(url: str, dest_path: Path):
    if dest_path.exists():
        return
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    # Download index files if not present
    download_file(FAISS_INDEX_URL, FAISS_INDEX_PATH)
    download_file(PKL_INDEX_URL, PKL_INDEX_PATH)
    
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        index_name="index",
        allow_dangerous_deserialization=True,
    )

@st.cache_resource(show_spinner=False)
def setup_qa_chain():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192"
    )

    vectorstore = load_vectorstore()

    prompt_template = """You are a helpful medical assistant. Use the following context to answer the question.

Context: {context}
Question: {question}

Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def main():
    st.title("🩺 HealthMate: Your Medical Buddy")
    query = st.text_input("Ask me anything!")

    if query:
        qa = setup_qa_chain()
        with st.spinner("🔍 Searching...
