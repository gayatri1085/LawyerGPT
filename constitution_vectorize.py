from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
import os

def vectorize_constitution():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Set OPENAI_API_KEY environment variable")

    pdf_path = "constitution.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError("constitution.pdf not found")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)

    with open("constitution_faiss.pkl", "wb") as f:
        pickle.dump(db, f)

    print("Vector DB created successfully.")

if __name__ == "__main__":
    vectorize_constitution()
