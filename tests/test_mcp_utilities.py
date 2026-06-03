import json
import unittest

from mcp_servers.utilities import handle_request


class MCPUtilitiesTest(unittest.TestCase):
    """Проверяет локальные MCP utility-инструменты."""

    def call_tool(self, name, arguments):
        """Вызывает инструмент напрямую через JSON-RPC handler."""
        response = handle_request({
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        return json.loads(response["content"][0]["text"])

    def test_tools_list_contains_expected_tools(self):
        """Сервер должен публиковать набор utility-инструментов."""
        response = handle_request({"method": "tools/list", "params": {}})
        tool_names = {tool["name"] for tool in response["tools"]}

        self.assertIn("text.count", tool_names)
        self.assertIn("json.validate", tool_names)
        self.assertIn("hash.sha256", tool_names)
        self.assertIn("random.uuid", tool_names)
        self.assertEqual(len(tool_names), 10)

    def test_text_count(self):
        """text.count считает базовую статистику текста."""
        payload = self.call_tool("text.count", {"text": "hello world"})

        self.assertTrue(payload["success"])
        self.assertEqual(payload["words"], 2)
        self.assertEqual(payload["characters"], 11)

    def test_json_validate(self):
        """json.validate отличает корректный JSON от некорректного."""
        valid = self.call_tool("json.validate", {"text": '{"a": 1}'})
        invalid = self.call_tool("json.validate", {"text": "{bad"})

        self.assertTrue(valid["valid"])
        self.assertFalse(invalid["valid"])

    def test_hash_sha256(self):
        """hash.sha256 возвращает ожидаемый SHA-256 для строки."""
        payload = self.call_tool("hash.sha256", {"text": "abc"})

        self.assertEqual(payload["hash"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
