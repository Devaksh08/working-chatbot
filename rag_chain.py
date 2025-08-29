from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def build_rag_chain(retriever, model="llama3.2"):
    llm = Ollama(model=model)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
