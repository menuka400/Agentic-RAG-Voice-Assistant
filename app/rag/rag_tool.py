import os
from langchain_core.tools import tool
from app.rag.retriever import get_retriever

def get_search_documents_tool():
    try:
        retriever = get_retriever()
        
        @tool("search_documents")
        def search_documents(query: str) -> str:
            """
            Searches and returns relevant information from uploaded documents and PDFs. Use this tool when the user asks questions about specific documents, internal knowledge, or uploaded context. Do NOT use this tool for general web search, current events, or internet info.
            """
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in uploaded documents."
            
            return "\n\n".join(
                f"Document snippet (Source: {os.path.basename(str(doc.metadata.get('source', 'Unknown')))}, Page: {doc.metadata.get('page', 'Unknown')}): {doc.page_content}" 
                for doc in docs
            )
            
        return search_documents
    except Exception as e:
        # Fallback tool if ingest has not been run or vectorstore is missing
        @tool("search_documents")
        def fallback_search_documents(query: str) -> str:
            """
            Searches and returns relevant information from uploaded documents and PDFs.
            """
            return f"Error: Document search is unavailable. Reason: {str(e)}. Please inform the user."
            
        return fallback_search_documents
