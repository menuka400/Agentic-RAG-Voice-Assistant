"""
app/tools.py — Agent tools.

Isolated tool definitions so more can be added easily (e.g., RAG later).
"""

from langchain_core.tools import tool
from ddgs import DDGS
from pydantic import BaseModel, Field
from datetime import datetime
import zoneinfo

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to look up on the web.")

@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web for real-time information, current events, recent news, or live data.
    CRITICAL: You MUST trigger this tool for any real-time or live queries such as "current time", "current date", live data, breaking news, or ongoing events.
    DO NOT trigger this tool on static factual or explanatory questions (e.g., "what timezone does Sri Lanka use" can be answered directly, but "what time is it in Sri Lanka now" requires a search).
    """
    try:
        # Perform the search using duckduckgo_search directly
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "No results found. (Search engine returned empty)."
            
        # Format results into a single string
        formatted_results = "\n\n".join(
            f"Title: {res.get('title')}\nSnippet: {res.get('body')}\nURL: {res.get('href')}"
            for res in results
        )
        return formatted_results
    except Exception as e:
        # Graceful fallback on search failure (e.g. rate limit, network issue)
        # Agent will receive this string and can explain the failure to the user.
        return f"Search failed due to an error: {str(e)}. Please inform the user."

class DateTimeInput(BaseModel):
    timezone: str = Field(description="The IANA timezone name (e.g., 'Asia/Colombo', 'America/New_York', 'UTC').")

@tool("get_current_datetime", args_schema=DateTimeInput)
def get_current_datetime(timezone: str) -> str:
    """
    Get the actual current date and time for a specific location.
    Use this tool for ANY questions asking "what is the current time/date in [place]".
    DO NOT use web_search for current time or date questions.
    Input must be a valid IANA timezone string (e.g., 'Asia/Colombo').
    """
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        current_time = datetime.now(tz)
        return f"The current date and time in {timezone} is: {current_time.strftime('%Y-%m-%d %I:%M:%S %p %Z')}"
    except Exception as e:
        return f"Error: Could not find timezone '{timezone}'. Please provide a valid IANA timezone name like 'Asia/Tokyo' or 'Europe/London'."

from app.rag.rag_tool import get_search_documents_tool
search_documents = get_search_documents_tool()
