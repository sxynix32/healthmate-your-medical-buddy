import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from pathlib import Path
from langchain.llms import Groq
from langchain.embeddings import GroqEmbeddings

# Path to your FAISS index files in current directory
INDEX_PATH = Path(".")

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return GroqEmbeddings()

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        faiss_index_name="index",     # basename without extension .faiss
        faiss_pickle_name="index.pkl",
        allow_dangerous_deserialization=True,
    )
    return vectorstore

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = Groq()  # Your Groq LLM instance (make sure env var GROQ_API_KEY is set)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def main():
    st.title("HealthMate - Your Medical Buddy with Groq")

    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")
    if query:
        with st.spinner("Getting answer from Groq..."):
            response = qa.run(query)
            st.write(response)

if __name__ == "__main__":
    main()
