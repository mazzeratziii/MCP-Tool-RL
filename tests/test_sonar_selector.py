import unittest

from src.selection.adaptive_selector import CandidateTool
from src.selection.sonar_selector import SonarToolSelector


class SonarToolSelectorTest(unittest.TestCase):
    def test_prefers_network_quality_when_semantics_are_close(self):
        selector = SonarToolSelector(semantic_threshold=0.5)
        candidates = [
            CandidateTool(
                name="Weather.Slow",
                semantic_score=0.82,
                available=True,
                latency=1.0,
                stability=0.7,
                estimated_success_rate=0.8,
            ),
            CandidateTool(
                name="Weather.Fast",
                semantic_score=0.80,
                available=True,
                latency=0.1,
                stability=1.0,
                estimated_success_rate=1.0,
            ),
        ]

        result = selector.select("weather in ufa", candidates)

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Weather.Fast")

    def test_returns_none_when_no_candidate_passes_threshold(self):
        selector = SonarToolSelector(semantic_threshold=0.9)

        result = selector.select(
            "weather",
            [CandidateTool(name="Weather.Low", semantic_score=0.5)],
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
