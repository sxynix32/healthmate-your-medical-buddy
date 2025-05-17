import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import Groq
from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load .env variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment.")
    st.stop()

# Initialize LLM and Embeddings with your API key
llm = Groq(api_key=GROQ_API_KEY)
embeddings = GroqEmbeddings(api_key=GROQ_API_KEY)

# Define index path & name (adjust if needed)
INDEX_PATH = "."  # your index files in root folder
INDEX_NAME = "index"  # 'index.faiss' and 'index.pkl' expected

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    return FAISS.load_local(INDEX_PATH, embeddings, index_name=INDEX_NAME)

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def main():
    st.title("HealthMate - Groq + FAISS + LangChain")

    qa_chain = setup_qa_chain()

    user_query = st.text_input("Ask me a medical question:")
    if user_query:
        response = qa_chain.run(user_query)
        st.write(response)

if __name__ == "__main__":
    main()


