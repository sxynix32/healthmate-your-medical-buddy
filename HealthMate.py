import streamlit as st
import requests
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Your Groq API key in .env as GROQ_API_KEY=...
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Your Hugging Face repo URLs for index files (replace with your actual URLs)
HF_INDEX_FAISS_URL = "https://huggingface.co/datasets/SxyNix344/healthmate/blob/main/index.faiss"
HF_INDEX_PKL_URL = "https://huggingface.co/datasets/SxyNix344/healthmate/blob/main/index.pkl"

# Optional HF token if repo is private; set in .env as HF_TOKEN=...
HF_TOKEN = os.getenv("HF_TOKEN")

# Local paths to save downloaded index files
LOCAL_INDEX_FAISS = Path("index.faiss")
LOCAL_INDEX_PKL = Path("index.pkl")

def download_file(url: str, dest_path: Path):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    if not dest_path.exists():
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return dest_path

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    # Download the index files if not present
    download_file(HF_INDEX_FAISS_URL, LOCAL_INDEX_FAISS)
    download_file(HF_INDEX_PKL_URL, LOCAL_INDEX_PKL)

    # Load HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load the FAISS vectorstore from downloaded files
    vectorstore = FAISS.load_local(
        ".",  # current directory where downloaded files are
        embeddings,
        index_name=None,  # default index_name is None, matches 'index.faiss'
        allow_dangerous_deserialization=True,
    )
    return vectorstore

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
        with st.spinner("🔍 Searching..."):
            response = qa.run(query)
        st.success(response)

if __name__ == "__main__":
    main()
