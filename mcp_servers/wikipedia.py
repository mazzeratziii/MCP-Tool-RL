"""
Wikipedia MCP Server
Provides Wikipedia search and article retrieval without API keys
"""
import json
import sys
from typing import Any, Dict
import requests


def search_wikipedia(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search Wikipedia articles"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = []

        if len(data) >= 4:
            titles = data[1]
            descriptions = data[2]
            urls = data[3]

            for i in range(len(titles)):
                results.append({
                    "title": titles[i],
                    "description": descriptions[i] if i < len(descriptions) else "",
                    "url": urls[i] if i < len(urls) else ""
                })

        return {
            "success": True,
            "error": None,
            "results": results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


def get_article(title: str) -> Dict[str, Any]:
    """Get Wikipedia article content"""
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        if pages:
            page = list(pages.values())[0]
            extract = page.get("extract", "")

            return {
                "success": True,
                "error": None,
                "title": page.get("title", ""),
                "content": extract[:1000]  # First 1000 chars
            }

        return {
            "success": False,
            "error": "Article not found",
            "title": "",
            "content": ""
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "title": "",
            "content": ""
        }


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP request"""
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "wikipedia.search",
                    "description": "Search Wikipedia articles",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results", "default": 5}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "wikipedia.get_article",
                    "description": "Get Wikipedia article content",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Article title"}
                        },
                        "required": ["title"]
                    }
                }
            ]
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "wikipedia.search":
            query = arguments.get("query", "")
            limit = arguments.get("limit", 5)
            result = search_wikipedia(query, limit)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ]
            }

        elif tool_name == "wikipedia.get_article":
            title = arguments.get("title", "")
            result = get_article(title)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ]
            }

    return {"error": "Unknown method"}


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()
