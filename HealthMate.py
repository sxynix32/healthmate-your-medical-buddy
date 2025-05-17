import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
import os

# Load environment variables
load_dotenv()

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
DATASET_ID = "SxyNix344/healthmate"

# Log in to Hugging Face if token is provided
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)

# Load HuggingFace embeddings
def load_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError as e:
        st.error("Missing required libraries. Please install 'transformers' and 'sentence-transformers'.")
        raise e

# Download and load FAISS index
@st.cache_resource
def load_vectorstore():
    st.info("🔽 Downloading vectorstore from Hugging Face...")
    index_file = hf_hub_download(repo_id=DATASET_ID, filename="index.faiss", token=HUGGINGFACE_TOKEN)
    pkl_file = hf_hub_download(repo_id=DATASET_ID, filename="index.pkl", token=HUGGINGFACE_TOKEN)
    
    embeddings = load_embeddings()
    return FAISS.load_local(
        folder_path=os.path.dirname(index_file),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

# Set up LLM and QA chain
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

# Streamlit app
def main():
    st.title("🩺 HealthMate: Your Medical Buddy")
    query = st.text_input("Ask me anything about your health:")

    if query:
        qa = setup_qa_chain()
        with st.spinner("🔍 Searching..."):
            response = qa.run(query)
        st.success(response)

if __name__ == "__main__":
    main()
