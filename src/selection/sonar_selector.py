from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .adaptive_selector import CandidateTool, SelectionResult, _clamp


@dataclass(frozen=True)
class SonarWeights:
    semantic: float = 0.70
    network: float = 0.30
    success: float = 0.40
    stability: float = 0.30
    latency: float = 0.20
    availability: float = 0.10


class SonarToolSelector:
    """
    SONAR-style baseline из статьи NetMCP.

    Политика намеренно оставлена простой и прозрачной:

        final_score = alpha * semantic_score + beta * network_score

    В отличие от AdaptiveToolSelector, этот baseline не использует intent
    matching, пересечение action/entity, группировку провайдеров и обучаемую
    обратную связь. Он нужен для сравнения semantic-only, SONAR-style routing
    и GRPO-политики.
    """

    def __init__(
        self,
        *,
        semantic_threshold: float = 0.55,
        weights: SonarWeights = SonarWeights(),
    ) -> None:
        """Initialize the object."""
        self.semantic_threshold = semantic_threshold
        self.weights = weights

    def select(
        self,
        query: str,
        candidates: Sequence[CandidateTool],
        *,
        exclude: Optional[Iterable[str]] = None,
    ) -> Optional[SelectionResult]:
        """Select select."""
        del query
        excluded = set(exclude or [])
        filtered = [
            tool for tool in candidates
            if tool.name not in excluded and tool.semantic_score >= self.semantic_threshold
        ]
        if not filtered:
            return None

        max_latency = max(max(0.0, tool.latency) for tool in filtered) or 1.0
        ranked = []
        for tool in filtered:
            parts = self.score_candidate(tool, max_latency=max_latency)
            ranked.append((parts["total"], tool, parts))

        ranked.sort(key=lambda item: item[0], reverse=True)
        score, tool, parts = ranked[0]
        snapshot = [
            {
                "rank": float(index + 1),
                "name": candidate.name,
                "score": round(score_parts["total"], 6),
                "semantic": round(score_parts["semantic"], 6),
                "functional": round(score_parts["semantic"], 6),
                "query_intent": None,
                "tool_intent": None,
                "functional_adjustment": 0.0,
                "rerank_adjustment": 0.0,
                "action_overlap": 0.0,
                "provider_overlap": 0.0,
                "entity_overlap": 0.0,
                "generic_penalty": 0.0,
                "group_adjustment": 0.0,
                "qos": round(score_parts["network"], 6),
                "latency_penalty": round(score_parts["latency_penalty"], 6),
                "retry_penalty": 0.0,
                "available": 1.0 if candidate.available else 0.0,
            }
            for index, (_, candidate, score_parts) in enumerate(ranked)
        ]

        return SelectionResult(
            tool_name=tool.name,
            score=score,
            rank=1,
            candidates=snapshot,
            reason=self._reason(tool, parts),
        )

    def score_candidate(self, tool: CandidateTool, *, max_latency: float) -> Dict[str, float]:
        """Score score candidate."""
        semantic = _clamp(tool.semantic_score)
        latency_ratio = _clamp(max(0.0, tool.latency) / max(max_latency, 1e-9))
        latency_quality = 1.0 - latency_ratio
        success = _clamp(tool.estimated_success_rate)
        stability = _clamp(min(tool.stability, 1.0 - max(0.0, tool.jitter)))
        availability = 1.0 if tool.available else 0.0

        network = _clamp(
            self.weights.success * success
            + self.weights.stability * stability
            + self.weights.latency * latency_quality
            + self.weights.availability * availability
        )
        total = self.weights.semantic * semantic + self.weights.network * network
        if not tool.available:
            total -= 1.0

        return {
            "total": total,
            "semantic": semantic,
            "network": network,
            "success": success,
            "stability": stability,
            "availability": availability,
            "latency_quality": latency_quality,
            "latency_penalty": 1.0 - latency_quality,
        }

    def _reason(self, tool: CandidateTool, parts: Mapping[str, float]) -> str:
        """Build explanation for reason."""
        return (
            f"{tool.name}: semantic={parts['semantic']:.3f}, "
            f"network={parts['network']:.3f}, available={tool.available}, "
            f"latency={tool.latency:.3f}s, stability={tool.stability:.3f}, "
            f"score={parts['total']:.3f}"
        )
