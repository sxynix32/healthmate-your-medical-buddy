import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY environment variable is missing.")
    st.stop()

# Initialize LLM and Embeddings
llm = Groq(api_key=GROQ_API_KEY)
embeddings = GroqEmbeddings(api_key=GROQ_API_KEY)

# Path to your FAISS index and pkl files
INDEX_PATH = "."  # current folder
INDEX_NAME = "index"

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(INDEX_PATH, embeddings, index_name=INDEX_NAME)

@st.cache_resource
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def main():
    st.title("HealthMate - Groq + FAISS + LangChain")

    qa_chain = setup_qa_chain()

    query = st.text_input("Ask a medical question:")
    if query:
        response = qa_chain.run(query)
        st.write(response)

if __name__ == "__main__":
    main()

