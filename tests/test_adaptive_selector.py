import unittest

from src.selection.adaptive_selector import (
    AdaptiveToolSelector,
    CandidateTool,
    ToolExecutionFeedback,
)
from src.selection.intent import FunctionalToolMatcher, ToolIntent


class AdaptiveToolSelectorTest(unittest.TestCase):
    def test_filters_out_semantically_wrong_tools(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6)

        result = selector.select(
            "find weather",
            [
                CandidateTool(name="Fast.ButWrong", semantic_score=0.2, latency=0.01),
                CandidateTool(name="Weather.Get", semantic_score=0.75, latency=0.2),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Weather.Get")

    def test_qos_can_break_tie_between_relevant_tools(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6)

        result = selector.select(
            "translate text",
            [
                CandidateTool(name="Translator.Slow", semantic_score=0.90, latency=0.9),
                CandidateTool(name="Translator.Fast", semantic_score=0.84, latency=0.1),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Translator.Fast")

    def test_observed_failures_reduce_future_rank(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6)
        selector.update_many(
            [
                ToolExecutionFeedback(tool_name="Search.Fast", success=False, latency=0.1, retries=1),
                ToolExecutionFeedback(tool_name="Search.Fast", success=False, latency=0.2, retries=1),
                ToolExecutionFeedback(tool_name="Search.Fast", success=False, latency=0.2, retries=2),
            ]
        )

        result = selector.select(
            "search documents",
            [
                CandidateTool(name="Search.Fast", semantic_score=0.86, latency=0.1),
                CandidateTool(name="Search.Reliable", semantic_score=0.82, latency=0.15),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Search.Reliable")

    def test_broad_weather_query_prefers_current_weather_over_humidity(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6)

        result = selector.select(
            "What is the weather in London?",
            [
                CandidateTool(name="Cloud Cast.Get humidty", semantic_score=0.74, latency=0.15),
                CandidateTool(
                    name="Weather.Current Weather Data of a location.",
                    semantic_score=0.73,
                    latency=0.15,
                    metadata={"description": "Current weather data for a location"},
                ),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Weather.Current Weather Data of a location.")

    def test_intent_matcher_tags_weather_forecast(self):
        matcher = FunctionalToolMatcher()

        query_intent = matcher.infer_query_intent("Give me the weather forecast for London tomorrow")
        tool_intent = matcher.infer_tool_intent(
            "Easy Weather.Daily forecast (5 days)",
            {"description": "Daily forecast for a location"},
        )

        self.assertEqual(query_intent, ToolIntent.WEATHER_FORECAST)
        self.assertEqual(tool_intent, ToolIntent.WEATHER_FORECAST)

    def test_adaptive_snapshot_contains_intents(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6)

        result = selector.select(
            "Give me the weather forecast for London tomorrow",
            [
                CandidateTool(
                    name="Easy Weather.Daily forecast (5 days)",
                    semantic_score=0.75,
                    metadata={"description": "Daily forecast for a location"},
                )
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.candidates[0]["query_intent"], "weather_forecast")
        self.assertEqual(result.candidates[0]["tool_intent"], "weather_forecast")


if __name__ == "__main__":
    unittest.main()
