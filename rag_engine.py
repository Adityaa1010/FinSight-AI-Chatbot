"""
rag_engine.py
=============
Full RAG pipeline for the AI Banking Chatbot.
LangChain 1.x compatible. Uses Groq API for LLM (free & fast).

Pipeline Steps:
  1. Load PDF documents
  2. Split them into chunks
  3. Create embeddings
  4. Store them in a vector database (Chroma)
  5. Retrieve relevant context
  6. Generate answers using an LLM (Groq)
"""

import os
from typing import List, Optional, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_groq import ChatGroq

from utils import log_step, get_env_variable


# ─────────────────────────────────────────────
# STEP 1: Load PDF Documents
# ─────────────────────────────────────────────
def load_documents(pdf_path: str) -> List[Document]:
    """
    Step 1: Load a PDF file and return a list of LangChain Document objects.
    Each page of the PDF becomes one Document.
    """
    log_step(1, "Loading PDF documents")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"   ✅ Loaded {len(documents)} page(s) from '{pdf_path}'")
    return documents


# ─────────────────────────────────────────────
# STEP 2: Split Documents into Chunks
# ─────────────────────────────────────────────
def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Step 2: Split large documents into smaller overlapping chunks.
    Smaller chunks = better retrieval precision.
    chunk_overlap ensures context is not lost at boundaries.
    """
    log_step(2, "Splitting documents into chunks")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    print(f"   ✅ Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ─────────────────────────────────────────────
# STEP 3: Create Embeddings
# ─────────────────────────────────────────────
def create_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Step 3: Load a HuggingFace embedding model (runs locally, no API needed).
    Converts text into numerical vectors for similarity search.
    all-MiniLM-L6-v2 is lightweight, fast, and accurate.
    """
    log_step(3, "Creating embeddings model")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"   ✅ Embedding model loaded: '{model_name}'")
    return embeddings


# ─────────────────────────────────────────────
# STEP 4: Store in Vector Database (Chroma)
# ─────────────────────────────────────────────
def store_in_vectordb(chunks: List[Document], embeddings: HuggingFaceEmbeddings, persist_dir: str = "./chroma_db") -> Chroma:
    """
    Step 4: Store chunks + their embeddings in Chroma (local vector DB).
    Persists to disk so we do not re-embed on every run.
    """
    log_step(4, "Storing chunks in Chroma vector database")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print(f"   ✅ Stored {len(chunks)} chunks in Chroma at '{persist_dir}'")
    return vectordb


# ─────────────────────────────────────────────
# STEP 5: Retrieve Relevant Context
# ─────────────────────────────────────────────
def create_retriever(vectordb: Chroma, k: int = 3):
    """
    Step 5: Create a retriever that finds the top-k most relevant chunks
    for any query using cosine similarity.
    """
    log_step(5, "Setting up retriever")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    print(f"   ✅ Retriever ready (top-{k} chunks per query)")
    return retriever


# ─────────────────────────────────────────────
# STEP 6: Generate Answers Using an LLM
# ─────────────────────────────────────────────
def create_llm() -> ChatGroq:
    """
    Step 6a: Initialize Groq LLM.
    Groq is free, ultra-fast, and reliable — perfect for demos and interviews.
    Model: llama-3.3-70b-versatile — powerful open-source model via Groq.
    Get your free API key at: https://console.groq.com
    """
    log_step(6, "Initializing LLM (Groq - free & fast)")

    api_key = get_env_variable("GROQ_API_KEY")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.3,
        max_tokens=512,
    )

    print("   ✅ LLM ready: llama-3.3-70b-versatile (Groq)")
    return llm


def format_docs(docs: List[Document]) -> str:
    """Helper: join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever, llm):
    """
    Step 6b: Build RAG chain using LCEL (LangChain Expression Language).
    Flow: question → retrieve context → fill prompt → LLM → parse answer
    """
    log_step(6, "Building LCEL RAG chain")

    prompt = PromptTemplate(
        template="""You are a helpful and professional AI banking assistant.
Use ONLY the context below to answer the customer's question.
If the answer is not in the context, say: "I am sorry, I do not have that information. Please contact our support team."

Context:
{context}

Customer Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    # LCEL chain — modern LangChain 1.x style
    rag_chain = (
        RunnableMap({
            "context": retriever | format_docs,   # retrieve → format as string
            "question": RunnablePassthrough()      # pass question unchanged
        })
        | prompt              # fill prompt template
        | llm                 # call Groq LLM
        | StrOutputParser()   # extract plain string from response
    )

    print("   ✅ RAG chain built (LCEL)!\n")
    return rag_chain


# ─────────────────────────────────────────────
# MASTER FUNCTION: Build Full RAG Pipeline
# ─────────────────────────────────────────────
def build_rag_pipeline(pdf_path: str):
    """
    Orchestrates all 6 RAG steps in order.
    Returns (rag_chain, retriever).
    """
    print("\n" + "="*50)
    print("  🏦  Building RAG Pipeline for Banking Chatbot")
    print("="*50 + "\n")

    documents  = load_documents(pdf_path)               # Step 1
    chunks     = split_documents(documents)              # Step 2
    embeddings = create_embeddings()                     # Step 3
    vectordb   = store_in_vectordb(chunks, embeddings)   # Step 4
    retriever  = create_retriever(vectordb)              # Step 5
    llm        = create_llm()                            # Step 6a
    qa_chain   = build_qa_chain(retriever, llm)          # Step 6b

    print("="*50)
    print("  ✅  RAG Pipeline Ready!")
    print("="*50 + "\n")

    return qa_chain, retriever

    """
rag_engine.py
=============
RAG Pipeline for the AI Banking Chatbot.
LangChain 1.x | Groq LLM | HuggingFace Embeddings | ChromaDB

Pipeline Steps:
  1. Load PDF documents
  2. Split into chunks
  3. Create embeddings (local, no API needed)
  4. Store in Chroma vector DB (reuses existing DB if already built)
  5. Retrieve relevant context
  6. Generate answers using Groq LLM
"""

