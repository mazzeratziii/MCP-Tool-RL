import unittest

from src.selection.healthcheck import HealthCheckItem, run_healthcheck


class HealthCheckTest(unittest.TestCase):
    def test_healthcheck_returns_items(self):
        items = run_healthcheck()

        self.assertTrue(items)
        self.assertTrue(all(isinstance(item, HealthCheckItem) for item in items))
        self.assertTrue(any(item.name == "MCP config" for item in items))


if __name__ == "__main__":
    unittest.main()
