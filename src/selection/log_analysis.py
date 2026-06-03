from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass
class ErrorExample:
    query: str
    selected_tool: Optional[str]
    relevant_tools: List[str]
    top3: List[str]
    reason: str
    query_intent: Optional[str] = None
    tool_intent: Optional[str] = None


@dataclass
class LogAnalysisSummary:
    total: int = 0
    correct: int = 0
    errors: int = 0
    success_count: int = 0
    target_in_top3_count: int = 0
    soft_correct_count: int = 0
    rerank_opportunity_count: int = 0
    avg_latency: float = 0.0
    avg_reward: float = 0.0
    error_reasons: Counter = field(default_factory=Counter)
    intent_pairs: Counter = field(default_factory=Counter)
    examples: Dict[str, List[ErrorExample]] = field(default_factory=lambda: defaultdict(list))

    @property
    def relevance_at_1(self) -> float:
        """Handle relevance at 1."""
        return self.correct / max(1, self.total)

    @property
    def relevance_at_3(self) -> float:
        """Handle relevance at 3."""
        return self.target_in_top3_count / max(1, self.total)

    @property
    def soft_relevance_at_1(self) -> float:
        """Handle soft relevance at 1."""
        return self.soft_correct_count / max(1, self.total)

    @property
    def success_rate(self) -> float:
        """Handle success rate."""
        return self.success_count / max(1, self.total)


class SelectionLogAnalyzer:
    def __init__(self, *, example_limit: int = 3, near_miss_threshold: float = 0.88) -> None:
        """Initialize the object."""
        self.example_limit = example_limit
        self.near_miss_threshold = near_miss_threshold

    def analyze_file(self, path: str) -> LogAnalysisSummary:
        """Analyze analyze file."""
        with open(path, "r", encoding="utf-8") as handle:
            return self.analyze_rows(json.loads(line) for line in handle if line.strip())

    def analyze_rows(self, rows: Iterable[Mapping[str, object]]) -> LogAnalysisSummary:
        """Analyze analyze rows."""
        summary = LogAnalysisSummary()
        latency_total = 0.0
        reward_total = 0.0

        for row in rows:
            summary.total += 1
            is_relevant = bool(row.get("is_relevant", False))
            selected = _as_optional_str(row.get("selected_tool"))
            relevant = [str(item) for item in row.get("relevant_tools", []) if item]
            top3 = [str(item) for item in row.get("top3", []) if item]
            query_intent = _as_optional_str(row.get("query_intent"))
            tool_intent = _as_optional_str(row.get("tool_intent"))

            if is_relevant:
                summary.correct += 1
                summary.soft_correct_count += 1
            else:
                summary.errors += 1
                reason = self.classify_error(
                    selected_tool=selected,
                    relevant_tools=relevant,
                    top3=top3,
                    query_intent=query_intent,
                    tool_intent=tool_intent,
                    success=bool(row.get("success", False)),
                    semantic_score=float(row.get("semantic_score", 0.0)),
                    functional_score=float(row.get("functional_score", 0.0)),
                )
                summary.error_reasons[reason] += 1
                if reason == "near_duplicate_or_alias":
                    summary.soft_correct_count += 1
                if reason == "right_tool_in_top3_not_top1":
                    summary.rerank_opportunity_count += 1
                if len(summary.examples[reason]) < self.example_limit:
                    summary.examples[reason].append(
                        ErrorExample(
                            query=str(row.get("query", "")),
                            selected_tool=selected,
                            relevant_tools=relevant[:5],
                            top3=top3[:3],
                            reason=reason,
                            query_intent=query_intent,
                            tool_intent=tool_intent,
                        )
                    )

            if bool(row.get("success", False)):
                summary.success_count += 1
            if bool(row.get("target_in_top3", False)):
                summary.target_in_top3_count += 1
            latency_total += float(row.get("latency", 0.0))
            reward_total += float(row.get("reward", 0.0))

            if query_intent or tool_intent:
                summary.intent_pairs[f"{query_intent}->{tool_intent}"] += 1

        n = max(1, summary.total)
        summary.avg_latency = latency_total / n
        summary.avg_reward = reward_total / n
        return summary

    def classify_error(
        self,
        *,
        selected_tool: Optional[str],
        relevant_tools: List[str],
        top3: List[str],
        query_intent: Optional[str],
        tool_intent: Optional[str],
        success: bool,
        semantic_score: float,
        functional_score: float,
    ) -> str:
        """Handle classify error."""
        if not selected_tool:
            return "no_selection"
        if self._is_near_miss(selected_tool, relevant_tools):
            return "near_duplicate_or_alias"
        if set(relevant_tools).intersection(top3):
            return "right_tool_in_top3_not_top1"
        if query_intent and tool_intent and query_intent != tool_intent:
            return "intent_mismatch"
        if not success:
            return "execution_failure"
        if semantic_score < 0.6 and functional_score < 0.6:
            return "retrieval_low_confidence"
        if self._looks_too_specific(selected_tool, relevant_tools):
            return "selected_too_specific"
        if self._looks_too_generic(selected_tool, relevant_tools):
            return "selected_too_generic"
        return "retrieval_miss_or_label_gap"

    def _is_near_miss(self, selected_tool: str, relevant_tools: List[str]) -> bool:
        """Handle is near miss."""
        selected_norm = _normalize_tool_name(selected_tool)
        for relevant in relevant_tools:
            relevant_norm = _normalize_tool_name(relevant)
            if selected_norm == relevant_norm:
                return True
            if SequenceMatcher(None, selected_norm, relevant_norm).ratio() >= self.near_miss_threshold:
                return True
        return False

    def _looks_too_specific(self, selected_tool: str, relevant_tools: List[str]) -> bool:
        """Handle looks too specific."""
        selected_tokens = _tokens(selected_tool)
        if not selected_tokens:
            return False
        for relevant in relevant_tools:
            relevant_tokens = _tokens(relevant)
            if relevant_tokens and selected_tokens > relevant_tokens:
                return True
        narrow_terms = {"id", "count", "humidity", "metadata", "status", "single", "specific"}
        return bool(selected_tokens & narrow_terms)

    def _looks_too_generic(self, selected_tool: str, relevant_tools: List[str]) -> bool:
        """Handle looks too generic."""
        selected_tokens = _tokens(selected_tool)
        for relevant in relevant_tools:
            relevant_tokens = _tokens(relevant)
            if selected_tokens and relevant_tokens and selected_tokens < relevant_tokens:
                return True
        generic_terms = {"search", "lookup", "list", "get", "all"}
        return bool(selected_tokens & generic_terms) and len(selected_tokens) <= 3


def print_analysis(summary: LogAnalysisSummary) -> None:
    """Print print analysis."""
    print("\n" + "=" * 60)
    print("Selection Log Analysis")
    print("=" * 60)
    print(f"Episodes:       {summary.total}")
    print(f"Correct:        {summary.correct} ({summary.relevance_at_1:.2%})")
    print(f"Soft Correct:   {summary.soft_correct_count} ({summary.soft_relevance_at_1:.2%})")
    print(f"Rerank cases:   {summary.rerank_opportunity_count}")
    print(f"Errors:         {summary.errors}")
    print(f"Relevance@3:    {summary.relevance_at_3:.2%}")
    print(f"Success rate:   {summary.success_rate:.2%}")
    print(f"Avg latency:    {summary.avg_latency:.3f}s")
    print(f"Avg reward:     {summary.avg_reward:.3f}")

    print("\nError reasons:")
    if not summary.error_reasons:
        print("  none")
    for reason, count in summary.error_reasons.most_common():
        print(f"  {reason}: {count}")

    print("\nIntent pairs:")
    for pair, count in summary.intent_pairs.most_common(10):
        print(f"  {pair}: {count}")

    if summary.examples:
        print("\nExamples:")
        for reason, examples in summary.examples.items():
            print(f"\n  {reason}:")
            for example in examples:
                print(f"    Query: {_safe_console(example.query[:180])}")
                print(f"    Selected: {_safe_console(example.selected_tool)}")
                print(f"    Relevant: {_safe_console(', '.join(example.relevant_tools[:3]))}")
                print(f"    Top3: {_safe_console(', '.join(example.top3))}")
                if example.query_intent or example.tool_intent:
                    print(f"    Intent: {_safe_console(example.query_intent)}->{_safe_console(example.tool_intent)}")


def _normalize_tool_name(value: str) -> str:
    """Normalize normalize tool name."""
    value = value.lower()
    value = value.replace("_v2", " 2").replace("api_v2", "api 2")
    value = value.replace("charater", "character")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _tokens(value: str) -> set:
    """Return tokens for tokens."""
    return set(re.findall(r"[a-z0-9]+", _normalize_tool_name(value)))


def _as_optional_str(value: object) -> Optional[str]:
    """Handle as optional str."""
    if value is None:
        return None
    return str(value)


def _safe_console(value: object) -> str:
    """Handle safe console."""
    if value is None:
        return ""
    text = str(value)
    return text.encode("ascii", errors="backslashreplace").decode("ascii")
