import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def main():
    base_dir = Path(__file__).parent
    pdf_dir = base_dir / "data" / "pdfs"
    db_dir = base_dir / "vectorstore"
    collection_name = "rag_chatbot"

    print(f"Scanning for PDFs in {pdf_dir}...")
    
    if not pdf_dir.exists():
        print(f"Directory {pdf_dir} does not exist. Creating it...")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some PDF files and run again.")
        return

    loader = PyPDFDirectoryLoader(str(pdf_dir))
    docs = loader.load()

    if not docs:
        print("No PDFs found in the directory.")
        return
        
    print(f"Loaded {len(docs)} pages from PDFs.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} text chunks.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    # Must run on CPU to avoid complex GPU setup requirements
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    if db_dir.exists():
        print(f"Vector store already exists at {db_dir}. Skipping ingest.")
        print("To force rebuild, delete the vectorstore/ folder.")
        return

    print("Generating embeddings and saving to ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_dir),
        collection_name=collection_name
    )
    db.persist()

    print(f"Successfully stored {len(chunks)} chunks in local ChromaDB at {db_dir}")

if __name__ == "__main__":
    main()
