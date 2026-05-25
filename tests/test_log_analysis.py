import unittest

from src.selection.log_analysis import SelectionLogAnalyzer


class SelectionLogAnalyzerTest(unittest.TestCase):
    def test_counts_basic_metrics_and_top3_errors(self):
        analyzer = SelectionLogAnalyzer()

        summary = analyzer.analyze_rows(
            [
                {
                    "query": "q1",
                    "selected_tool": "Weather.Current",
                    "relevant_tools": ["Weather.Current"],
                    "top3": ["Weather.Current"],
                    "is_relevant": True,
                    "target_in_top3": True,
                    "success": True,
                    "latency": 0.1,
                    "reward": 4.0,
                    "query_intent": "current_weather",
                    "tool_intent": "current_weather",
                },
                {
                    "query": "q2",
                    "selected_tool": "Weather.Forecast",
                    "relevant_tools": ["Weather.Current"],
                    "top3": ["Weather.Forecast", "Weather.Current"],
                    "is_relevant": False,
                    "target_in_top3": True,
                    "success": True,
                    "latency": 0.3,
                    "reward": 0.5,
                    "query_intent": "current_weather",
                    "tool_intent": "weather_forecast",
                },
            ]
        )

        self.assertEqual(summary.total, 2)
        self.assertEqual(summary.correct, 1)
        self.assertEqual(summary.errors, 1)
        self.assertEqual(summary.error_reasons["right_tool_in_top3_not_top1"], 1)
        self.assertEqual(summary.rerank_opportunity_count, 1)
        self.assertAlmostEqual(summary.relevance_at_1, 0.5)
        self.assertAlmostEqual(summary.relevance_at_3, 1.0)
        self.assertAlmostEqual(summary.soft_relevance_at_1, 0.5)

    def test_detects_near_duplicate_alias(self):
        analyzer = SelectionLogAnalyzer()

        reason = analyzer.classify_error(
            selected_tool="Throne of Glass API_v2.Get Charater by ID",
            relevant_tools=["Throne of Glass API 2.Get Character by ID"],
            top3=[],
            query_intent="data_lookup",
            tool_intent="data_lookup",
            success=True,
            semantic_score=0.8,
            functional_score=0.9,
        )

        self.assertEqual(reason, "near_duplicate_or_alias")

    def test_near_duplicate_counts_as_soft_correct(self):
        analyzer = SelectionLogAnalyzer()

        summary = analyzer.analyze_rows(
            [
                {
                    "query": "q",
                    "selected_tool": "Throne of Glass API_v2.Get Charater by ID",
                    "relevant_tools": ["Throne of Glass API 2.Get Character by ID"],
                    "top3": [],
                    "is_relevant": False,
                    "target_in_top3": False,
                    "success": True,
                    "latency": 0.1,
                    "reward": 0.5,
                    "query_intent": "data_lookup",
                    "tool_intent": "data_lookup",
                }
            ]
        )

        self.assertEqual(summary.correct, 0)
        self.assertEqual(summary.soft_correct_count, 1)
        self.assertAlmostEqual(summary.soft_relevance_at_1, 1.0)


if __name__ == "__main__":
    unittest.main()
