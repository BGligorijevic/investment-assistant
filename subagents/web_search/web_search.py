from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import BaseTool

def get_web_search_tool(max_results: int = 3) -> BaseTool:
    """
    Returns a configured web search tool.

    This function acts as an abstraction layer. Currently, it uses Tavily,
    but it can be easily modified to return a different search tool
    (e.g., from Google, Perplexity, etc.) without changing the main agent logic.

    Args:
        max_results: The maximum number of search results to return.
    """
    return TavilySearchResults(max_results=max_results)

