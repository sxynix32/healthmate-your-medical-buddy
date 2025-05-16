import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_PATH = "faiss_index"

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

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
    query = st.text_input("Ask me anything!:")
    
    if query:
        qa = setup_qa_chain()
        with st.spinner("🔍 Searching..."):
            response = qa.run(query)
        st.success(response)

if __name__ == "__main__":
    main()
