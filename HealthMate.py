import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI  # or your preferred LLM
from pathlib import Path

INDEX_PATH = Path(".")  # current directory where index.faiss & index.pkl are
INDEX_NAME = "index.faiss"
PICKLE_NAME = "index.pkl"

@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        faiss_index_name="index",
        faiss_pickle_name="index.pkl",
        allow_dangerous_deserialization=True,
    )
    return vectorstore


@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = OpenAI(temperature=0)  # Replace with your LLM config
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    st.title("HealthMate Medical Buddy")

    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")
    if query:
        with st.spinner("Generating answer..."):
            response = qa.run(query)
        st.markdown(f"**Answer:** {response}")

if __name__ == "__main__":
    main()
