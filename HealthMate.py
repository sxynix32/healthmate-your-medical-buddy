import os
import requests

class GroqLLM:
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, prompt):
        url = "https://api.groq.ai/v1/completions"  # Example Groq endpoint
        headers = {"Authorization": f"Bearer {self.api_key}"}
        json_data = {
            "model": "gpt-4o-mini",  # Or your Groq model name
            "prompt": prompt,
            "max_tokens": 150,
        }
        response = requests.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["text"]

# Usage in your Streamlit app:
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import GroqEmbeddings
from pathlib import Path

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
        faiss_index_name="index",
        faiss_pickle_name="index.pkl",
        allow_dangerous_deserialization=True,
    )
    return vectorstore

@st.cache_resource(show_spinner=True)
def setup_qa_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = GroqLLM(groq_api_key)

    # Simple retrieval + generation manually:
    def qa_chain(query):
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {query}\nAnswer:"
        return llm(prompt)

    return qa_chain

def main():
    st.title("HealthMate - Your Medical Buddy with Groq (Manual)")

    qa = setup_qa_chain()

    query = st.text_input("Ask your medical question:")
    if query:
        with st.spinner("Getting answer from Groq..."):
            response = qa(query)
            st.write(response)

if __name__ == "__main__":
    main()
