import unittest

from src.selection.adaptive_selector import AdaptiveToolSelector, CandidateTool
from src.selection.tool_features import ToolFeatureExtractor


class ToolFeatureExtractorTest(unittest.TestCase):
    def test_action_overlap_rewards_requested_details(self):
        extractor = ToolFeatureExtractor()

        features = extractor.score(
            "Find iPhones and provide product details and price",
            "Ebay.Product Details",
            {"description": "Get product details including title, condition, and price"},
        )

        self.assertGreater(features.action_overlap, 0.0)
        self.assertGreater(features.adjustment, 0.0)

    def test_generic_search_gets_penalty_for_specific_request(self):
        extractor = ToolFeatureExtractor()

        features = extractor.score(
            "Find iPhones and provide product details and price",
            "iOS Store.Search",
            {"description": "Search apps in the iOS store"},
        )

        self.assertGreater(features.generic_penalty, 0.0)

    def test_selector_prefers_specific_action_over_generic_search(self):
        selector = AdaptiveToolSelector(semantic_threshold=0.6, rerank_weight=1.0)

        result = selector.select(
            "Can you search for iPhones in Germany and provide product details including title and price?",
            [
                CandidateTool(
                    name="iOS Store.Search",
                    semantic_score=0.82,
                    metadata={"description": "Search apps in the iOS store"},
                ),
                CandidateTool(
                    name="Ebay.Product Details",
                    semantic_score=0.78,
                    metadata={"description": "Get product details, title, condition, and price"},
                ),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Ebay.Product Details")

    def test_selector_boosts_provider_mentioned_in_query(self):
        selector = AdaptiveToolSelector(
            semantic_threshold=0.6,
            provider_group_weight=1.0,
        )

        result = selector.select(
            "Search for office chairs on Tokopedia and show titles, prices, images, and ratings.",
            [
                CandidateTool(
                    name="Ikea API.Search By Keyword Filters",
                    semantic_score=0.82,
                    metadata={"description": "Search furniture products by keyword"},
                ),
                CandidateTool(
                    name="TokopediaApi.Search Product",
                    semantic_score=0.78,
                    metadata={"description": "Search Tokopedia products by keyword"},
                ),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "TokopediaApi.Search Product")
        self.assertGreater(result.candidates[0]["group_adjustment"], 0.0)

    def test_generic_provider_token_does_not_override_named_tool(self):
        selector = AdaptiveToolSelector(
            semantic_threshold=0.6,
            provider_group_weight=1.0,
        )

        result = selector.select(
            "Please check the server status of the Waifu tool and get the user metadata.",
            [
                CandidateTool(
                    name="User demo.getUsers",
                    semantic_score=0.84,
                    metadata={"description": "Get users"},
                ),
                CandidateTool(
                    name="Waifu.Check server status",
                    semantic_score=0.80,
                    metadata={"description": "Check Waifu server status"},
                ),
            ],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "Waifu.Check server status")


if __name__ == "__main__":
    unittest.main()
