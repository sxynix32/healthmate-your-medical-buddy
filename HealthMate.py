import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from pathlib import Path

# Set your index path - current directory
INDEX_PATH = Path(".")  

# Cache embeddings to avoid reload
@st.cache_resource(show_spinner=True)
def load_embeddings():
    return OpenAIEmbeddings()

# Cache vectorstore load
@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = load_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        faiss_index_name="index",      # basename without extension
        faiss_pickle_name="index.pkl",
        allow_dangerous_deserialization=True,
    )
    return vectorstore

# Cache QA chain setup
@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa_chain

def main():
    st.title("HealthMate - Your Medical Buddy")

    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")
    if query:
        with st.spinner("Searching..."):
            response = qa.run(query)
            st.write(response)

if __name__ == "__main__":
    main()
