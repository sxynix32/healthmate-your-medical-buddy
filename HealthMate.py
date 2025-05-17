import streamlit as st
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Replace these URLs with your Hugging Face raw file URLs
HF_INDEX_FAISS_URL = "https://huggingface.co/your-username/your-repo/resolve/main/index.faiss"
HF_INDEX_PKL_URL = "https://huggingface.co/your-username/your-repo/resolve/main/index.pkl"

LOCAL_INDEX_FAISS = "./index.faiss"
LOCAL_INDEX_PKL = "./index.pkl"

def download_file(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"Downloaded {local_path}")
    else:
        print(f"{local_path} already exists, skipping download")

@st.cache_resource(show_spinner=False)
def load_vectorstore():
    download_file(HF_INDEX_FAISS_URL, LOCAL_INDEX_FAISS)
    download_file(HF_INDEX_PKL_URL, LOCAL_INDEX_PKL)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = FAISS.load_local(
        ".",             # directory where the files live
        embeddings,
        index_name="index",  # basename of your .faiss and .pkl files (without extension)
        allow_dangerous_deserialization=True,
    )
    return vectorstore

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

