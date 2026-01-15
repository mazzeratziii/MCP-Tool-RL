from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class MCPRequest:
    tool_name: str
    query: str

@dataclass
class MCPResponse:
    success: bool
    result: Optional[Any]
    error: Optional[str]
    latency: float
