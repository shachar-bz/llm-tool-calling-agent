from duckduckgo_search import DDGS

WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web with DuckDuckGo for fresh external information that is "
            "not present in the provided resource files. Returns the top 3 results "
            "as dicts with title, snippet, and url."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                },
            },
            "required": ["query"],
        },
    },
}


def web_search(query: str) -> list:
    try:
        results = DDGS().text(query, max_results=3)
        return [
            {
                "title": result.get("title"),
                "snippet": result.get("body"),
                "url": result.get("href"),
            }
            for result in results
        ]
    except Exception as exception:
        return [{"error": str(exception)}]
