import os
from dotenv import load_dotenv
import streamlit as st
from langchain.llms import Groq
from langchain.embeddings import GroqEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load env variables from .env
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in environment variables!")
    st.stop()

# Initialize Groq LLM and embeddings with your API key set in env
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

def load_vectorstore(embeddings):
    """Load or create FAISS vectorstore."""
    if os.path.exists("faiss_index.pkl") and os.path.exists("faiss_index.faiss"):
        # Load existing FAISS index
        return FAISS.load_local(".", embeddings)
    else:
        # Example: load documents, split, embed, and build vectorstore
        docs = TextLoader("docs/your_docs.txt").load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs_split = splitter.split_documents(docs)
        return FAISS.from_documents(docs_split, embeddings)

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    embeddings = GroqEmbeddings()
    vectorstore = load_vectorstore(embeddings)
    retriever = vectorstore.as_retriever()
    llm = Groq(model="llama2", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    st.title("HealthMate - Medical Buddy with Groq LLM")
    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")
    if query:
        with st.spinner("Fetching answer..."):
            answer = qa.run(query)
        st.markdown(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()



