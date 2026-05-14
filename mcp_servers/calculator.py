"""
Simple Calculator MCP Server
Provides basic mathematical operations without requiring API keys
"""
import json
import sys
from typing import Any, Dict


def calculate(expression: str) -> Dict[str, Any]:
    """Safely evaluate mathematical expression"""
    try:
        # Безопасное вычисление (только математические операции)
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return {
                "success": False,
                "error": "Invalid characters in expression",
                "result": None
            }

        result = eval(expression, {"__builtins__": {}}, {})
        return {
            "success": True,
            "error": None,
            "result": float(result)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "result": None
        }


def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP request"""
    method = request.get("method")
    params = request.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "calculator.evaluate",
                    "description": "Evaluate mathematical expressions",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            ]
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "calculator.evaluate":
            expression = arguments.get("expression", "")
            result = calculate(expression)
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
    # Simple JSON-RPC server
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = handle_request(request)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()
