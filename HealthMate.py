import os
import streamlit as st
from langchain.llms import Groq
from langchain.embeddings import GroqEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

# Constants - update if needed
INDEX_PATH = "."  # current folder where index.faiss and index.pkl are located
INDEX_NAME = "index"  # without extension

# Load Groq API key from env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

# Setup Groq embeddings and LLM
embeddings = GroqEmbeddings()
llm = Groq()

@st.cache_resource(show_spinner=True)
def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        folder_path=INDEX_PATH,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever: BaseRetriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Optional: customize prompt or use default
    prompt_template = """Use the following context to answer the question.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain

def main():
    st.title("HealthMate — Your Medical Buddy")

    qa = setup_qa_chain()

    user_question = st.text_input("Ask your medical question:")

    if user_question:
        with st.spinner("Searching answer..."):
            answer = qa.run(user_question)
        st.markdown("**Answer:**")
        st.write(answer)

if __name__ == "__main__":
    main()
