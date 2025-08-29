import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(file_paths):
    docs = []
    for path in file_paths:
        if os.path.getsize(path) == 0:
            print(f"⚠️ Skipping empty file: {path}")
            continue

        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        else:
            print(f"⚠️ Unsupported file format: {path}")
            continue

        docs.extend(loader.load())
    return docs

def split_documents(documents, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(documents)
