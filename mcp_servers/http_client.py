"""
HTTP Client MCP Server
Provides HTTP GET/POST requests without API keys
"""
import json
import sys
from typing import Any, Dict
import requests


def http_get(url: str, headers: Dict[str, str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Make HTTP GET request"""
    try:
        response = requests.get(url, headers=headers or {}, timeout=timeout)

        return {
            "success": True,
            "error": None,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:5000],  # First 5000 chars
            "content_type": response.headers.get("Content-Type", "")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": 0,
            "headers": {},
            "body": "",
            "content_type": ""
        }


def http_post(url: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout: int = 10) -> Dict[str, Any]:
    """Make HTTP POST request"""
    try:
        response = requests.post(url, json=data or {}, headers=headers or {}, timeout=timeout)

        return {
            "success": True,
            "error": None,
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:5000],
            "content_type": response.headers.get("Content-Type", "")
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": 0,
            "headers": {},
            "body": "",
            "content_type": ""
        }


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP request"""
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "http.get",
                    "description": "Make HTTP GET request",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to request"},
                            "headers": {"type": "object", "description": "HTTP headers"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 10}
                        },
                        "required": ["url"]
                    }
                },
                {
                    "name": "http.post",
                    "description": "Make HTTP POST request",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to request"},
                            "data": {"type": "object", "description": "JSON data to send"},
                            "headers": {"type": "object", "description": "HTTP headers"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 10}
                        },
                        "required": ["url"]
                    }
                }
            ]
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "http.get":
            url = arguments.get("url", "")
            headers = arguments.get("headers", {})
            timeout = arguments.get("timeout", 10)
            result = http_get(url, headers, timeout)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result)
                    }
                ]
            }

        elif tool_name == "http.post":
            url = arguments.get("url", "")
            data = arguments.get("data", {})
            headers = arguments.get("headers", {})
            timeout = arguments.get("timeout", 10)
            result = http_post(url, data, headers, timeout)
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
