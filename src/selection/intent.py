from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Set


class ToolIntent(str, Enum):
    CURRENT_WEATHER = "current_weather"
    WEATHER_FORECAST = "weather_forecast"
    HISTORICAL_WEATHER = "historical_weather"
    WEATHER_METRIC = "weather_metric"
    CALCULATION = "calculation"
    TRANSLATION = "translation"
    SEARCH = "search"
    DATA_LOOKUP = "data_lookup"
    NO_TOOL_NEEDED = "no_tool_needed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IntentMatch:
    query_intent: ToolIntent
    tool_intent: ToolIntent
    adjustment: float
    reasons: Set[str] = field(default_factory=set)


class FunctionalToolMatcher:
    """
    Правиловый matcher функционального соответствия.

    Он намеренно небольшой и прозрачный: преобразует метаданные ToolBench в
    intent-теги и применяет бонусы/штрафы перед QoS-ранжированием. Позже его
    можно заменить классификатором, обученным на метках ToolBench, сохранив
    тот же публичный API.
    """

    def infer_query_intent(self, query: str) -> ToolIntent:
        """Infer infer query intent."""
        q = query.lower()

        if self._looks_like_calculation(q):
            return ToolIntent.CALCULATION
        if any(term in q for term in ("translate", "translation", "convert language", "переведи", "перевод")):
            return ToolIntent.TRANSLATION
        if any(term in q for term in ("hello", "hi ", "thanks", "thank you", "привет", "спасибо")):
            return ToolIntent.NO_TOOL_NEEDED
        if self._is_weather_query(q):
            if any(term in q for term in ("forecast", "tomorrow", "next week", "next 5 days", "coming days")):
                return ToolIntent.WEATHER_FORECAST
            if any(term in q for term in ("historical", "history", "past weather", "yesterday", "last week")):
                return ToolIntent.HISTORICAL_WEATHER
            if any(term in q for term in ("humidity", "wind", "pressure", "uv", "rain", "precipitation")):
                return ToolIntent.WEATHER_METRIC
            return ToolIntent.CURRENT_WEATHER
        if any(term in q for term in ("search", "find", "lookup", "look up", "wiki", "wikipedia", "найди", "поиск")):
            return ToolIntent.SEARCH
        return ToolIntent.DATA_LOOKUP

    def infer_tool_intent(self, name: str, metadata: Mapping[str, object]) -> ToolIntent:
        """Infer infer tool intent."""
        text = self.candidate_text(name, metadata)

        if "notoolneeded" in text or "no external tool" in text:
            return ToolIntent.NO_TOOL_NEEDED
        if any(term in text for term in ("calculator", "calculate", "arithmetic", "math expression")):
            return ToolIntent.CALCULATION
        if any(term in text for term in ("translate", "translation", "language translation")):
            return ToolIntent.TRANSLATION

        if "weather" in text or any(term in text for term in ("humidity", "forecast", "meteorological", "temperature")):
            if any(term in text for term in ("historical", "history", "dailyweather", "past weather")):
                return ToolIntent.HISTORICAL_WEATHER
            if any(term in text for term in ("forecast", "daily forecast", "hourly forecast", "5 days", "5-day")):
                return ToolIntent.WEATHER_FORECAST
            if any(term in text for term in ("humidity", "humidty", "wind", "pressure", "uv index", "air quality")):
                return ToolIntent.WEATHER_METRIC
            if any(term in text for term in ("current weather", "weather data", "current conditions")):
                return ToolIntent.CURRENT_WEATHER
            return ToolIntent.CURRENT_WEATHER

        if any(term in text for term in ("search", "find", "lookup", "wikipedia", "web search")):
            return ToolIntent.SEARCH
        return ToolIntent.DATA_LOOKUP

    def match(self, query: str, name: str, metadata: Mapping[str, object]) -> IntentMatch:
        """Match match."""
        query_intent = self.infer_query_intent(query)
        tool_intent = self.infer_tool_intent(name, metadata)
        reasons: Set[str] = set()
        adjustment = 0.0

        if query_intent == tool_intent:
            adjustment += 0.16
            reasons.add("intent_match")
        elif self._compatible_weather_intents(query_intent, tool_intent):
            adjustment += 0.06
            reasons.add("compatible_weather_intent")
        elif self._compatible_lookup_intents(query_intent, tool_intent):
            adjustment += 0.03
            reasons.add("compatible_lookup_intent")
        elif query_intent != ToolIntent.UNKNOWN and tool_intent != ToolIntent.UNKNOWN:
            adjustment -= 0.14
            reasons.add("intent_mismatch")

        if query_intent == ToolIntent.CURRENT_WEATHER and tool_intent in {
            ToolIntent.WEATHER_METRIC,
            ToolIntent.HISTORICAL_WEATHER,
        }:
            adjustment -= 0.12
            reasons.add("too_narrow_for_broad_weather")

        text = self.candidate_text(name, metadata)
        if query_intent == ToolIntent.CURRENT_WEATHER and "location" in text:
            adjustment += 0.04
            reasons.add("location_supported")
        if query_intent == ToolIntent.WEATHER_FORECAST and "forecast" in text:
            adjustment += 0.04
            reasons.add("forecast_supported")

        return IntentMatch(
            query_intent=query_intent,
            tool_intent=tool_intent,
            adjustment=adjustment,
            reasons=reasons,
        )

    def candidate_text(self, name: str, metadata: Mapping[str, object]) -> str:
        """Handle candidate text."""
        parts = [name]
        for value in metadata.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(item) for item in value[:8])
        return " ".join(parts).lower()

    def _compatible_weather_intents(self, query_intent: ToolIntent, tool_intent: ToolIntent) -> bool:
        """Handle compatible weather intents."""
        weather_intents = {
            ToolIntent.CURRENT_WEATHER,
            ToolIntent.WEATHER_FORECAST,
            ToolIntent.HISTORICAL_WEATHER,
            ToolIntent.WEATHER_METRIC,
        }
        return query_intent in weather_intents and tool_intent in weather_intents

    def _compatible_lookup_intents(self, query_intent: ToolIntent, tool_intent: ToolIntent) -> bool:
        """Handle compatible lookup intents."""
        lookup_intents = {ToolIntent.SEARCH, ToolIntent.DATA_LOOKUP}
        return query_intent in lookup_intents and tool_intent in lookup_intents

    def _is_weather_query(self, query: str) -> bool:
        """Handle is weather query."""
        return any(term in query for term in (
            "weather",
            "temperature",
            "forecast",
            "humidity",
            "wind",
            "погода",
            "температура",
            "прогноз",
            "влажность",
            "ветер",
            "давление",
        ))

    def _looks_like_calculation(self, query: str) -> bool:
        """Handle looks like calculation."""
        return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", query)) or any(
            term in query for term in ("calculate", "calculator", "sum of", "multiply", "divide", "рассчитай", "посчитай")
        )
