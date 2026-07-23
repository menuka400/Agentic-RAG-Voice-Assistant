from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_retriever():
    base_dir = Path(__file__).parent.parent.parent
    db_dir = base_dir / "vectorstore"
    collection_name = "rag_chatbot"

    if not db_dir.exists():
        raise RuntimeError(f"Vector store not found at {db_dir}. Please run ingest.py first.")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # Return top 5 relevant chunks
    return db.as_retriever(search_kwargs={"k": 5})
