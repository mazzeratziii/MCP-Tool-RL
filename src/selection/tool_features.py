from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Mapping, Set


ACTION_TOKENS = {
    "account",
    "all",
    "amount",
    "balance",
    "calculate",
    "cash",
    "category",
    "comments",
    "convert",
    "count",
    "details",
    "download",
    "flow",
    "forecast",
    "history",
    "income",
    "info",
    "list",
    "metadata",
    "order",
    "orders",
    "price",
    "prices",
    "product",
    "products",
    "profile",
    "radiation",
    "rates",
    "reviews",
    "search",
    "solar",
    "statement",
    "statistics",
    "status",
    "validate",
    "validation",
}

GENERIC_ACTIONS = {
    "get",
    "list",
    "search",
    "lookup",
    "all",
    "info",
}

STOPWORDS = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "api",
    "as",
    "be",
    "by",
    "can",
    "could",
    "for",
    "from",
    "give",
    "get",
    "help",
    "i",
    "in",
    "include",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "please",
    "provide",
    "show",
    "the",
    "to",
    "tool",
    "using",
    "with",
    "you",
    "в",
    "для",
    "и",
    "мне",
    "на",
    "по",
    "покажи",
    "у",
    "что",
}

TOKEN_TRANSLATIONS = {
    "погода": {"weather"},
    "температура": {"temperature", "weather"},
    "прогноз": {"forecast", "weather"},
    "ветер": {"wind", "weather"},
    "влажность": {"humidity", "weather"},
    "давление": {"pressure", "weather"},
    "москва": {"moscow"},
    "уфа": {"ufa"},
    "казань": {"kazan"},
    "найди": {"search", "find"},
    "поиск": {"search"},
    "переведи": {"translate"},
    "перевод": {"translation"},
    "рассчитай": {"calculate"},
    "посчитай": {"calculate"},
}


@dataclass(frozen=True)
class ToolTextFeatures:
    provider_tokens: Set[str]
    action_tokens: Set[str]
    all_tokens: Set[str]
    action_overlap: float
    entity_overlap: float
    provider_overlap: float
    generic_penalty: float
    alias_score: float
    adjustment: float


class ToolFeatureExtractor:
    """
    Domain-agnostic text features for reranking ToolBench-style tools.

    ToolBench names often look like "Provider.Action". Provider/action overlap
    is useful when multiple API families solve similar tasks but labels prefer
    a particular family.
    """

    def score(
        self,
        query: str,
        tool_name: str,
        metadata: Mapping[str, object],
    ) -> ToolTextFeatures:
        """Score score."""
        provider, action = split_tool_name(tool_name)
        provider_tokens = content_tokens(provider)
        action_tokens = content_tokens(action)
        metadata_tokens = content_tokens(_metadata_text(metadata))
        tool_tokens = provider_tokens | action_tokens | metadata_tokens
        query_tokens = content_tokens(query)

        query_actions = (query_tokens & ACTION_TOKENS) - GENERIC_ACTIONS
        action_overlap = _overlap(query_actions, action_tokens | metadata_tokens)
        provider_overlap = _overlap(query_tokens, provider_tokens)
        entity_overlap = _overlap(query_tokens - ACTION_TOKENS, provider_tokens | metadata_tokens)
        generic_penalty = self._generic_penalty(query_tokens, action_tokens)
        alias_score = self.alias_score(query, tool_name)

        adjustment = 0.0
        adjustment += 0.07 * action_overlap
        adjustment += 0.10 * provider_overlap
        adjustment += 0.05 * entity_overlap
        adjustment += 0.04 * alias_score
        adjustment -= generic_penalty

        return ToolTextFeatures(
            provider_tokens=provider_tokens,
            action_tokens=action_tokens,
            all_tokens=tool_tokens,
            action_overlap=action_overlap,
            entity_overlap=entity_overlap,
            provider_overlap=provider_overlap,
            generic_penalty=generic_penalty,
            alias_score=alias_score,
            adjustment=adjustment,
        )

    def alias_score(self, query: str, tool_name: str) -> float:
        """Handle alias score."""
        del query
        normalized = normalize_tool_name(tool_name)
        provider, action = split_tool_name(normalized)
        if not provider or not action:
            return 0.0
        return SequenceMatcher(None, provider, action).ratio() * 0.1

    def _generic_penalty(self, query_tokens: Set[str], action_tokens: Set[str]) -> float:
        """Handle generic penalty."""
        if not action_tokens:
            return 0.0
        specific_query_actions = (query_tokens & ACTION_TOKENS) - GENERIC_ACTIONS
        specific_tool_actions = (action_tokens & ACTION_TOKENS) - GENERIC_ACTIONS
        generic_action_ratio = len(action_tokens & GENERIC_ACTIONS) / max(1, len(action_tokens))
        if specific_tool_actions:
            return 0.0
        if specific_query_actions and generic_action_ratio >= 0.5:
            return 0.14
        if len(action_tokens) <= 2 and action_tokens & GENERIC_ACTIONS:
            return 0.04
        return 0.0


def split_tool_name(name: str) -> tuple[str, str]:
    """Split split tool name."""
    if "." not in name:
        return name, ""
    provider, action = name.split(".", 1)
    return provider, action


def normalize_tool_name(value: str) -> str:
    """Normalize normalize tool name."""
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = value.lower()
    value = value.replace("_v2", " 2").replace("api_v2", "api 2")
    value = value.replace("charater", "character")
    value = value.replace("roullette", "roulette")
    return " ".join(re.findall(r"[a-zа-яё0-9]+", value))


def content_tokens(value: str) -> Set[str]:
    """Handle content tokens."""
    normalized = normalize_tool_name(value)
    tokens = {
        token
        for token in re.findall(r"[a-zа-яё0-9]+", normalized)
        if len(token) > 1 and token not in STOPWORDS
    }
    expanded = set(tokens)
    for token in tokens:
        expanded.update(TOKEN_TRANSLATIONS.get(token, set()))
        expanded.update(_split_compound_token(token))
    return expanded


def _split_compound_token(token: str) -> Set[str]:
    """Split split compound token."""
    if token.endswith("api") and len(token) > 3:
        return {token[:-3]}
    known = {
        "dailyweather": {"daily", "weather"},
        "getcountries": {"countries"},
        "realtoragentlist": {"realtor", "agent", "list"},
    }
    if token in known:
        return known[token]
    return set()


def _metadata_text(metadata: Mapping[str, object]) -> str:
    """Handle metadata text."""
    parts = []
    for value in metadata.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(item) for item in value[:8])
    return " ".join(parts)


def _overlap(left: Set[str], right: Set[str]) -> float:
    """Handle overlap."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
