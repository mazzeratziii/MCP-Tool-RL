import unittest

from src.selection.intent import FunctionalToolMatcher, ToolIntent
from src.selection.query_normalization import expand_query_for_retrieval, is_too_ambiguous_query
from src.selection.tool_features import content_tokens


class QueryNormalizationTest(unittest.TestCase):
    def test_expands_russian_weather_query(self):
        expanded = expand_query_for_retrieval("погода уфа")

        self.assertIn("weather", expanded)
        self.assertIn("ufa", expanded)

    def test_expands_russian_weather_query_with_repeated_letter_typo(self):
        expanded = expand_query_for_retrieval("ПОогода уфа")

        self.assertIn("weather", expanded)
        self.assertIn("ufa", expanded)

    def test_russian_weather_intent(self):
        matcher = FunctionalToolMatcher()

        self.assertEqual(matcher.infer_query_intent("погода уфа"), ToolIntent.CURRENT_WEATHER)

    def test_russian_tokens_expand_to_english_features(self):
        tokens = content_tokens("погода уфа")

        self.assertIn("weather", tokens)
        self.assertIn("ufa", tokens)

    def test_short_uppercase_query_is_ambiguous(self):
        self.assertTrue(is_too_ambiguous_query("HQD"))
        self.assertFalse(is_too_ambiguous_query("audi a5 2026"))

    def test_expands_russian_nft_query(self):
        expanded = expand_query_for_retrieval("Топ 20 нфт")

        self.assertIn("nft", expanded)
        self.assertIn("ranking", expanded)

    def test_expands_russian_nft_query_with_letter_swap_typo(self):
        expanded = expand_query_for_retrieval("Топ 20 нтф")

        self.assertIn("nft", expanded)
        self.assertIn("ranking", expanded)

    def test_expands_russian_article_query(self):
        expanded = expand_query_for_retrieval("AI статьи 2026")

        self.assertIn("articles", expanded)
        self.assertIn("news", expanded)

    def test_expands_russian_product_query(self):
        expanded = expand_query_for_retrieval("сок добрый")

        self.assertIn("product", expanded)
        self.assertIn("dobry", expanded)


if __name__ == "__main__":
    unittest.main()
