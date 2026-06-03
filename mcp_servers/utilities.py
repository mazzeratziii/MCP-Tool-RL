"""
Utilities MCP Server
Provides small local text, JSON, hash, date/time and random tools without API keys
"""
import datetime
import hashlib
import json
import random
import re
import string
import sys
import uuid
from typing import Any, Dict


def count_text(text: str) -> Dict[str, Any]:
    """Count characters, words and lines in text"""
    try:
        words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
        lines = text.splitlines() or ([text] if text else [])

        return {
            "success": True,
            "error": None,
            "characters": len(text),
            "characters_no_spaces": len(text.replace(" ", "")),
            "words": len(words),
            "lines": len(lines)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "characters": 0,
            "characters_no_spaces": 0,
            "words": 0,
            "lines": 0
        }


def extract_keywords(text: str, limit: int = 10) -> Dict[str, Any]:
    """Extract frequent keywords from text"""
    try:
        stop_words = {
            "and", "or", "the", "a", "an", "of", "to", "in", "for", "on", "with",
            "и", "или", "в", "во", "на", "с", "со", "по", "для", "из", "к", "от"
        }
        words = [
            word.lower()
            for word in re.findall(r"\b[\w'-]{3,}\b", text, flags=re.UNICODE)
            if word.lower() not in stop_words
        ]

        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1

        keywords = [
            {"keyword": word, "count": count}
            for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max(1, limit)]
        ]

        return {
            "success": True,
            "error": None,
            "keywords": keywords
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "keywords": []
        }


def slugify_text(text: str) -> Dict[str, Any]:
    """Create URL-friendly slug from text"""
    try:
        normalized = re.sub(r"[^\w\s-]", "", text.lower(), flags=re.UNICODE)
        slug = re.sub(r"[\s_]+", "-", normalized).strip("-")

        return {
            "success": True,
            "error": None,
            "slug": slug
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "slug": ""
        }


def search_regex(text: str, pattern: str) -> Dict[str, Any]:
    """Search text with a regular expression"""
    try:
        matches = re.findall(pattern, text)

        return {
            "success": True,
            "error": None,
            "matches": matches,
            "count": len(matches)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "matches": [],
            "count": 0
        }


def get_current_datetime(timezone: str = "UTC") -> Dict[str, Any]:
    """Return current date and time"""
    try:
        if timezone.upper() in {"UTC", "Z"}:
            now = datetime.datetime.now(datetime.timezone.utc)
            timezone_name = "UTC"
        else:
            now = datetime.datetime.now()
            timezone_name = "local"

        return {
            "success": True,
            "error": None,
            "timezone": timezone_name,
            "iso": now.isoformat(),
            "timestamp": now.timestamp()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timezone": timezone,
            "iso": "",
            "timestamp": 0
        }


def validate_json(text: str) -> Dict[str, Any]:
    """Validate JSON text"""
    try:
        value = json.loads(text)

        return {
            "success": True,
            "error": None,
            "valid": True,
            "type": type(value).__name__
        }
    except json.JSONDecodeError as e:
        return {
            "success": True,
            "error": None,
            "valid": False,
            "message": str(e),
            "type": None
        }


def format_json(text: str, indent: int = 2) -> Dict[str, Any]:
    """Pretty-print JSON text"""
    try:
        value = json.loads(text)
        formatted = json.dumps(value, ensure_ascii=False, indent=max(0, min(indent, 8)))

        return {
            "success": True,
            "error": None,
            "formatted": formatted
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "formatted": ""
        }


def calculate_sha256(text: str) -> Dict[str, Any]:
    """Calculate SHA-256 hash for text"""
    try:
        return {
            "success": True,
            "error": None,
            "hash": hashlib.sha256(text.encode("utf-8")).hexdigest()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "hash": ""
        }


def generate_uuid() -> Dict[str, Any]:
    """Generate UUID4"""
    try:
        return {
            "success": True,
            "error": None,
            "uuid": str(uuid.uuid4())
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "uuid": ""
        }


def generate_password(length: int = 16) -> Dict[str, Any]:
    """Generate random password"""
    try:
        length = max(8, min(int(length), 128))
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = "".join(random.choice(alphabet) for _ in range(length))

        return {
            "success": True,
            "error": None,
            "password": password,
            "length": length
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "password": "",
            "length": 0
        }


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP request"""
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "text.count",
                    "description": "Count characters, words and lines in text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "text.extract_keywords",
                    "description": "Extract frequent keywords from text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text"},
                            "limit": {"type": "integer", "description": "Max keywords", "default": 10}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "text.slugify",
                    "description": "Create URL-friendly slug from text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "regex.search",
                    "description": "Search text with a regular expression",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text"},
                            "pattern": {"type": "string", "description": "Regular expression"}
                        },
                        "required": ["text", "pattern"]
                    }
                },
                {
                    "name": "datetime.now",
                    "description": "Return current date and time",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "timezone": {"type": "string", "description": "Timezone label", "default": "UTC"}
                        },
                        "required": []
                    }
                },
                {
                    "name": "json.validate",
                    "description": "Validate JSON text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "JSON text"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "json.format",
                    "description": "Pretty-print JSON text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "JSON text"},
                            "indent": {"type": "integer", "description": "Indent size", "default": 2}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "hash.sha256",
                    "description": "Calculate SHA-256 hash for text",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Input text"}
                        },
                        "required": ["text"]
                    }
                },
                {
                    "name": "random.uuid",
                    "description": "Generate UUID4",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "random.password",
                    "description": "Generate random password",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "length": {"type": "integer", "description": "Password length", "default": 16}
                        },
                        "required": []
                    }
                }
            ]
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "text.count":
            result = count_text(arguments.get("text", ""))
        elif tool_name == "text.extract_keywords":
            result = extract_keywords(arguments.get("text", ""), arguments.get("limit", 10))
        elif tool_name == "text.slugify":
            result = slugify_text(arguments.get("text", ""))
        elif tool_name == "regex.search":
            result = search_regex(arguments.get("text", ""), arguments.get("pattern", ""))
        elif tool_name == "datetime.now":
            result = get_current_datetime(arguments.get("timezone", "UTC"))
        elif tool_name == "json.validate":
            result = validate_json(arguments.get("text", ""))
        elif tool_name == "json.format":
            result = format_json(arguments.get("text", ""), arguments.get("indent", 2))
        elif tool_name == "hash.sha256":
            result = calculate_sha256(arguments.get("text", ""))
        elif tool_name == "random.uuid":
            result = generate_uuid()
        elif tool_name == "random.password":
            result = generate_password(arguments.get("length", 16))
        else:
            return {"error": "Unknown tool"}

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, ensure_ascii=False)
                }
            ]
        }

    return {"error": "Unknown method"}


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response, ensure_ascii=False))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}, ensure_ascii=False))
            sys.stdout.flush()
