import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import GroqEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # replace with your Groq LLM import if available

# Your Groq API key here
GROQ_API_KEY = "your_groq_api_key"

# Constants for index location
INDEX_PATH = "."
INDEX_NAME = "index"

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = GroqEmbeddings(groq_api_key=GROQ_API_KEY)

    st.write("Loading FAISS index from folder:", os.path.abspath(INDEX_PATH))
    st.write("Files found:", os.listdir(INDEX_PATH))

    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, index_name=INDEX_NAME)
    return vectorstore

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()

    # Use ChatOpenAI as a placeholder - replace with your Groq LLM chain if available
    llm = ChatOpenAI(temperature=0)  

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa

def main():
    st.title("HealthMate - Groq + FAISS Chatbot")

    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")

    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()


