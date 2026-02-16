class MessageType:
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    TOOLS_LIST = "tools/list/response"
    TOOL_RESULT = "tools/call/response"
    ERROR = "error"
    TOOL_REGISTERED = "notification/tool_registered"
    TOOL_UNREGISTERED = "notification/tool_unregistered"

PROTOCOL_VERSION = "2024-11-05"

class Status:
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"