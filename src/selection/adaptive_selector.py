from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .intent import FunctionalToolMatcher
from .tool_features import ToolFeatureExtractor, content_tokens, split_tool_name


GENERIC_PROVIDER_TOKENS = {
    "api",
    "data",
    "demo",
    "finance",
    "get",
    "info",
    "information",
    "market",
    "markets",
    "price",
    "prices",
    "real",
    "search",
    "service",
    "stock",
    "stats",
    "time",
    "tool",
    "user",
    "users",
    "video",
    "weather",
}


@dataclass(frozen=True)
class CandidateTool:
    name: str
    semantic_score: float
    available: bool = True
    latency: float = 0.0
    jitter: float = 0.0
    stability: float = 1.0
    estimated_success_rate: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionResult:
    tool_name: str
    score: float
    rank: int
    candidates: List[Dict[str, object]]
    reason: str


@dataclass(frozen=True)
class ToolExecutionFeedback:
    tool_name: str
    success: bool
    latency: float
    retries: int = 0
    semantic_score: Optional[float] = None


@dataclass
class _ObservedToolStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    retry_total: int = 0
    latency_ema: Optional[float] = None
    success_ema: Optional[float] = None

    def update(self, feedback: ToolExecutionFeedback, alpha: float) -> None:
        """Update update."""
        self.calls += 1
        self.retry_total += max(0, feedback.retries)
        if feedback.success:
            self.successes += 1
        else:
            self.failures += 1

        self.latency_ema = _ema(self.latency_ema, max(0.0, feedback.latency), alpha)
        self.success_ema = _ema(self.success_ema, 1.0 if feedback.success else 0.0, alpha)

    @property
    def retry_rate(self) -> float:
        """Handle retry rate."""
        if self.calls == 0:
            return 0.0
        return self.retry_total / self.calls

    @property
    def observed_success_rate(self) -> float:
        """Handle observed success rate."""
        if self.calls == 0:
            return 1.0
        return self.success_ema if self.success_ema is not None else self.successes / self.calls


class AdaptiveToolSelector:
    """
    Библиотечный алгоритм выбора инструментов.

    Он разделяет функциональные требования и нефункциональную адаптацию:
    1. semantic gate оставляет только инструменты, которые могут ответить на запрос;
    2. QoS score адаптирует ранжирование по latency, jitter/stability,
       availability, observed success rate и истории retry.

    Класс намеренно не зависит от Torch/LLM-кода, чтобы его можно было
    использовать в demo, тестах, benchmark и как baseline для RL-экспериментов.
    """

    def __init__(
        self,
        *,
        semantic_threshold: float = 0.55,
        semantic_weight: float = 0.58,
        success_weight: float = 0.20,
        latency_weight: float = 0.12,
        stability_weight: float = 0.07,
        retry_weight: float = 0.03,
        unavailable_penalty: float = 1.0,
        observation_alpha: float = 0.35,
        rerank_weight: float = 0.35,
        provider_group_weight: float = 0.0,
        functional_matcher: Optional[FunctionalToolMatcher] = None,
        feature_extractor: Optional[ToolFeatureExtractor] = None,
    ) -> None:
        """Initialize the object."""
        self.semantic_threshold = semantic_threshold
        self.semantic_weight = semantic_weight
        self.success_weight = success_weight
        self.latency_weight = latency_weight
        self.stability_weight = stability_weight
        self.retry_weight = retry_weight
        self.unavailable_penalty = unavailable_penalty
        self.observation_alpha = observation_alpha
        self.rerank_weight = rerank_weight
        self.provider_group_weight = provider_group_weight
        self.functional_matcher = functional_matcher or FunctionalToolMatcher()
        self.feature_extractor = feature_extractor or ToolFeatureExtractor()
        self._stats: Dict[str, _ObservedToolStats] = {}

    def select(
        self,
        query: str,
        candidates: Sequence[CandidateTool],
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[SelectionResult]:
        """Select select."""
        excluded = set(exclude or [])
        filtered = [
            tool for tool in candidates
            if tool.name not in excluded and tool.semantic_score >= self.semantic_threshold
        ]
        if not filtered:
            return None

        max_latency = max(max(0.0, tool.latency) for tool in filtered) or 1.0
        ranked = []
        mentioned_provider_tokens = self._mentioned_provider_tokens(query, filtered)
        for tool in filtered:
            score_parts = self.score_candidate(tool, query=query, max_latency=max_latency)
            group_adjustment = self._group_adjustment(query, tool, mentioned_provider_tokens)
            score_parts["group_adjustment"] = group_adjustment
            score_parts["functional"] = _clamp(score_parts["functional"] + group_adjustment)
            score_parts["semantic"] = self.semantic_weight * score_parts["functional"]
            score_parts["total"] = (
                score_parts["semantic"]
                + score_parts["qos"]
                - score_parts["latency_penalty"]
                - score_parts["retry_penalty"]
                - score_parts["unavailable_penalty"]
            )
            ranked.append((score_parts["total"], tool, score_parts))

        ranked.sort(key=lambda item: item[0], reverse=True)
        score, tool, score_parts = ranked[0]
        snapshot = [
            {
                "rank": float(i + 1),
                "name": candidate.name,
                "score": round(parts["total"], 6),
                "semantic": round(parts["raw_semantic"], 6),
                "functional": round(parts["functional"], 6),
                "query_intent": parts["query_intent"],
                "tool_intent": parts["tool_intent"],
                "functional_adjustment": round(parts["functional_adjustment"], 6),
                "rerank_adjustment": round(parts["rerank_adjustment"], 6),
                "action_overlap": round(parts["action_overlap"], 6),
                "provider_overlap": round(parts["provider_overlap"], 6),
                "entity_overlap": round(parts["entity_overlap"], 6),
                "generic_penalty": round(parts["generic_penalty"], 6),
                "group_adjustment": round(parts["group_adjustment"], 6),
                "qos": round(parts["qos"], 6),
                "latency_penalty": round(parts["latency_penalty"], 6),
                "retry_penalty": round(parts["retry_penalty"], 6),
                "available": 1.0 if candidate.available else 0.0,
            }
            for i, (_, candidate, parts) in enumerate(ranked)
        ]

        return SelectionResult(
            tool_name=tool.name,
            score=score,
            rank=1,
            candidates=snapshot,
            reason=self._reason(tool, score_parts),
        )

    def score_candidate(self, tool: CandidateTool, *, query: str, max_latency: float) -> Dict[str, float]:
        """Score score candidate."""
        stats = self._stats.get(tool.name, _ObservedToolStats())

        intent_match = self.functional_matcher.match(query, tool.name, tool.metadata)
        text_features = self.feature_extractor.score(query, tool.name, tool.metadata)
        functional_score = _clamp(
            tool.semantic_score
            + intent_match.adjustment
            + self.rerank_weight * text_features.adjustment
        )
        observed_success = stats.observed_success_rate
        success_rate = _clamp((tool.estimated_success_rate + observed_success) / 2.0)
        observed_latency = stats.latency_ema if stats.latency_ema is not None else tool.latency
        latency_ratio = _clamp(observed_latency / max(max_latency, 1e-9))
        stability = _clamp(min(tool.stability, 1.0 - max(0.0, tool.jitter)))
        retry_penalty = min(1.0, stats.retry_rate / 3.0)

        semantic = self.semantic_weight * functional_score
        qos = self.success_weight * success_rate + self.stability_weight * stability
        latency_penalty = self.latency_weight * latency_ratio
        retry_cost = self.retry_weight * retry_penalty
        unavailable_cost = self.unavailable_penalty if not tool.available else 0.0

        total = semantic + qos - latency_penalty - retry_cost - unavailable_cost
        return {
            "total": total,
            "raw_semantic": _clamp(tool.semantic_score),
            "semantic": semantic,
            "functional": functional_score,
            "query_intent": intent_match.query_intent.value,
            "tool_intent": intent_match.tool_intent.value,
            "functional_adjustment": intent_match.adjustment,
            "rerank_adjustment": self.rerank_weight * text_features.adjustment,
            "action_overlap": text_features.action_overlap,
            "provider_overlap": text_features.provider_overlap,
            "entity_overlap": text_features.entity_overlap,
            "generic_penalty": text_features.generic_penalty,
            "qos": qos,
            "latency_penalty": latency_penalty,
            "retry_penalty": retry_cost,
            "unavailable_penalty": unavailable_cost,
        }

    def update(self, feedback: ToolExecutionFeedback) -> None:
        """Update update."""
        stats = self._stats.setdefault(feedback.tool_name, _ObservedToolStats())
        stats.update(feedback, alpha=self.observation_alpha)

    def update_many(self, feedback_items: Iterable[ToolExecutionFeedback]) -> None:
        """Update update many."""
        for feedback in feedback_items:
            self.update(feedback)

    def choose_with_retries(
        self,
        query: str,
        candidates: Sequence[CandidateTool],
        execute: Callable[[str], ToolExecutionFeedback],
        *,
        max_retries: int = 2,
    ) -> Optional[SelectionResult]:
        """Handle choose with retries."""
        attempted: List[str] = []
        last_result: Optional[SelectionResult] = None

        for _ in range(max(1, max_retries + 1)):
            result = self.select(query, candidates, exclude=attempted)
            if result is None:
                return last_result

            last_result = result
            attempted.append(result.tool_name)
            feedback = execute(result.tool_name)
            self.update(feedback)

            if feedback.success:
                return result

        return last_result

    def get_observed_stats(self) -> Dict[str, Dict[str, float]]:
        """Return get observed stats."""
        return {
            name: {
                "calls": float(stats.calls),
                "success_rate": round(stats.observed_success_rate, 6),
                "avg_retries": round(stats.retry_rate, 6),
                "latency_ema": round(stats.latency_ema or 0.0, 6),
            }
            for name, stats in self._stats.items()
        }

    def _reason(self, tool: CandidateTool, score_parts: Mapping[str, float]) -> str:
        """Build explanation for reason."""
        return (
            f"{tool.name}: semantic={tool.semantic_score:.3f}, functional={score_parts['functional']:.3f}, "
            f"intent={score_parts['query_intent']}->{score_parts['tool_intent']}, "
            f"rerank={score_parts['rerank_adjustment']:.3f}, group={score_parts.get('group_adjustment', 0.0):.3f}, "
            f"available={tool.available}, latency={tool.latency:.3f}s, "
            f"stability={tool.stability:.3f}, score={score_parts['total']:.3f}"
        )

    def _mentioned_provider_tokens(self, query: str, candidates: Sequence[CandidateTool]) -> set:
        """Handle mentioned provider tokens."""
        query_tokens = content_tokens(query)
        mentioned = set()
        for tool in candidates:
            provider, _ = split_tool_name(tool.name)
            provider_tokens = self._specific_provider_tokens(provider)
            mentioned.update(query_tokens & provider_tokens)
        return mentioned

    def _group_adjustment(
        self,
        query: str,
        tool: CandidateTool,
        mentioned_provider_tokens: set,
    ) -> float:
        """Handle group adjustment."""
        del query
        if not mentioned_provider_tokens:
            return 0.0

        provider, _ = split_tool_name(tool.name)
        provider_tokens = self._specific_provider_tokens(provider)
        if provider_tokens & mentioned_provider_tokens:
            return 0.08 * self.provider_group_weight
        return -0.03 * self.provider_group_weight

    def _specific_provider_tokens(self, provider: str) -> set:
        """Handle specific provider tokens."""
        return {
            token
            for token in content_tokens(provider)
            if token not in GENERIC_PROVIDER_TOKENS and len(token) >= 4
        }


def candidates_from_environment_state(state: Mapping[str, object]) -> List[CandidateTool]:
    """Handle candidates from environment state."""
    tools = state.get("tools", [])
    if not isinstance(tools, list):
        return []

    candidates: List[CandidateTool] = []
    for tool in tools:
        if not isinstance(tool, Mapping):
            continue
        candidates.append(
            CandidateTool(
                name=str(tool.get("name", "")),
                semantic_score=float(tool.get("semantic_score", 0.0)),
                available=bool(tool.get("available", True)),
                latency=float(tool.get("latency", 0.0)),
                jitter=float(tool.get("jitter", 0.0)),
                stability=float(tool.get("stability", 1.0)),
                estimated_success_rate=float(tool.get("success_rate", 1.0)),
                metadata=tool,
            )
        )
    return candidates


def _ema(current: Optional[float], value: float, alpha: float) -> float:
    """Handle ema."""
    if current is None:
        return value
    return alpha * value + (1.0 - alpha) * current


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp clamp."""
    return max(low, min(high, value))
