import os
from src.document_loader import load_documents, split_documents
from src.vectorstore import create_vectorstore
from src.rag_chain import build_rag_chain

def main():
    # Load documents
    file_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith((".pdf", ".txt"))]
    docs = load_documents(file_paths)
    chunks = split_documents(docs)

    # Create retriever
    retriever = create_vectorstore(chunks)

    # Build RAG chain with LLaMA 3.2
    qa_chain = build_rag_chain(retriever, model="llama3.2")

    # Chat loop
    print("🤖 RAG Chatbot Ready! Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        if query.lower() == "exit":
            break
        answer = qa_chain.run(query)
        print("\n🤖 Answer:", answer)

if __name__ == "__main__":
    main()
