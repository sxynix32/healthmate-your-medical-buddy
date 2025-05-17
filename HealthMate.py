import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pathlib import Path
import requests

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Your Hugging Face repo info (edit these URLs)
HF_REPO = "your-hf-username/your-hf-repo-name"
HF_INDEX_FAISS_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/index.faiss"
HF_INDEX_PKL_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/index.pkl"

LOCAL_INDEX_DIR = Path("faiss_index")
LOCAL_INDEX_DIR.mkdir(exist_ok=True)
LOCAL_INDEX_FAISS = LOCAL_INDEX_DIR / "index.faiss"
LOCAL_INDEX_PKL = LOCAL_INDEX_DIR / "index.pkl"

@st.cache_resource(show_spinner=False)
def download_file(url: str, dest_path: Path):
    if not dest_path.exists():
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return dest_path

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    # Download files from Hugging Face if not exist
    download_file(HF_INDEX_FAISS_URL, LOCAL_INDEX_FAISS)
    download_file(HF_INDEX_PKL_URL, LOCAL_INDEX_PKL)

    embeddings = HuggingFaceEmbeddings()

    # Load the FAISS index from local files
    vectorstore = FAISS.load_local(
        str(LOCAL_INDEX_DIR),
        embeddings,
        index_name="index",
        allow_dangerous_deserialization=True,
    )
    return vectorstore

@st.cache_resource(show_spinner=False)
def setup_qa_chain():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama3-8b-8192",
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
        chain_type_kwargs={"prompt": prompt},
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

