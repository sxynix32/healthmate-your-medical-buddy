import os
import requests
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

# Your Groq API key from Streamlit secrets or env var
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

class GroqEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed(self, texts):
        # texts can be list or single string
        if isinstance(texts, str):
            texts = [texts]
        url = "https://api.groq.ai/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": "gpt-4o-mini", "input": texts}
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

def load_vectorstore():
    embeddings = GroqEmbeddings(GROQ_API_KEY)
    INDEX_PATH = "."  # folder containing index.faiss and index.pkl
    INDEX_NAME = "index"  # base filename without extensions
    print("Loading index from:", INDEX_PATH, "with base name:", INDEX_NAME)
    print("Files in directory:", os.listdir(INDEX_PATH))  # debug line
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, index_name=INDEX_NAME)
    return vectorstore


def query_groq(prompt):
    url = "https://api.groq.ai/v1/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    json_data = {
        "model": "gpt-4o-mini",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.3,
        "stop": None,
    }
    response = requests.post(url, headers=headers, json=json_data)
    response.raise_for_status()
    completion = response.json()
    return completion["choices"][0]["text"]

def main():
    st.title("HealthMate — Medical Chatbot with Groq")
    vectorstore = load_vectorstore()

    query = st.text_input("Ask me a medical question:")
    if query:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        answer = query_groq(prompt)
        st.write(answer)

if __name__ == "__main__":
    main()

