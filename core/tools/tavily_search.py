import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

def search_web(query: str) -> list[dict] | str:
    """
    Uses the Tavily API to search for destination info (attractions, safety, local tips, seasonal considerations).
    
    Args:
        query (str): The search query.
        
    Returns:
        list[dict] | str: A clean list of relevant results containing title, url, and content, or an error message.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY is not set."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=False
        )
        
        # Clean up results
        results = []
        for res in response.get("results", []):
            results.append({
                "title": res.get("title"),
                "url": res.get("url"),
                "content": res.get("content")
            })
            
        return results if results else "No relevant results found."
    except Exception as e:
        return f"Error occurred during Tavily search: {str(e)}"

if __name__ == "__main__":
    print("Testing search_web...")
    result = search_web("Top 3 attractions in Paris, France")
    print(result)
