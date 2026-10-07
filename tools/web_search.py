"""Web search via DuckDuckGo, for facts that are not in the task's resource files."""

from duckduckgo_search import DDGS

from .base import Tool

MAX_RESULTS = 3


def web_search(query: str) -> list[dict]:
    return [
        {
            "title": result.get("title"),
            "snippet": result.get("body"),
            "url": result.get("href"),
        }
        for result in DDGS().text(query, max_results=MAX_RESULTS)
    ]


WEB_SEARCH = Tool(
    name="web_search",
    description=(
        "Search the web with DuckDuckGo for fresh external information that is "
        "not present in the provided resource files. Returns the top 3 results "
        "as dicts with title, snippet, and url."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The web search query.",
            },
        },
        "required": ["query"],
    },
    handler=web_search,
)
