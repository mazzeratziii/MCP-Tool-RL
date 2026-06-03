from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional

from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode
from src.selection.adaptive_selector import (
    AdaptiveToolSelector,
    ToolExecutionFeedback,
    candidates_from_environment_state,
)
from src.selection.sonar_selector import SonarToolSelector


@dataclass
class BenchmarkSummary:
    policy: str
    episodes: int
    relevance_at_1: float
    relevance_at_3: float
    success_rate: float
    avg_latency: float
    avg_reward: float
    avg_retries: float
    log_path: Optional[str]


class SelectionBenchmark:
    def __init__(
        self,
        config: Config,
        *,
        network_mode: NetworkMode,
        top_k: int = 10,
        semantic_threshold: float = 0.55,
        rerank_weight: float = 0.35,
        provider_group_weight: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Initialize the object."""
        self.config = config
        self.network_mode = network_mode
        self.top_k = top_k
        self.semantic_threshold = semantic_threshold
        self.rerank_weight = rerank_weight
        self.provider_group_weight = provider_group_weight
        self.seed = seed

    def run(
        self,
        *,
        policy: str,
        episodes: int,
        log_path: Optional[str] = None,
    ) -> BenchmarkSummary:
        """Run run."""
        random.seed(self.seed)
        env = MCPEnvironment(self.config, network_mode=self.network_mode)
        selector = AdaptiveToolSelector(
            semantic_threshold=self.semantic_threshold,
            rerank_weight=self.rerank_weight,
            provider_group_weight=self.provider_group_weight,
        )
        sonar_selector = SonarToolSelector(semantic_threshold=self.semantic_threshold)
        prompts = self._sample_prompts(episodes)

        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

        totals = {
            "episodes": 0,
            "relevance_at_1": 0,
            "relevance_at_3": 0,
            "success": 0,
            "latency": 0.0,
            "reward": 0.0,
            "retries": 0,
        }

        log_file = open(log_path, "w", encoding="utf-8") if log_path else None
        try:
            for prompt in prompts:
                row = self._run_episode(env, selector, sonar_selector, prompt, policy)
                totals["episodes"] += 1
                totals["relevance_at_1"] += int(row["is_relevant"])
                totals["relevance_at_3"] += int(row["target_in_top3"])
                totals["success"] += int(row["success"])
                totals["latency"] += float(row["latency"])
                totals["reward"] += float(row["reward"])
                totals["retries"] += int(row["retries"])

                if log_file:
                    log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            if log_file:
                log_file.close()

        n = max(1, totals["episodes"])
        return BenchmarkSummary(
            policy=policy,
            episodes=totals["episodes"],
            relevance_at_1=totals["relevance_at_1"] / n,
            relevance_at_3=totals["relevance_at_3"] / n,
            success_rate=totals["success"] / n,
            avg_latency=totals["latency"] / n,
            avg_reward=totals["reward"] / n,
            avg_retries=totals["retries"] / n,
            log_path=log_path,
        )

    def _run_episode(
        self,
        env: MCPEnvironment,
        selector: AdaptiveToolSelector,
        sonar_selector: SonarToolSelector,
        prompt: Mapping[str, object],
        policy: str,
    ) -> Dict[str, object]:
        """Run run episode."""
        state = env.reset(dict(prompt))
        state["tools"] = state.get("tools", [])[:self.top_k]
        candidates = candidates_from_environment_state(state)
        relevant_names = {
            tool.get("name")
            for tool in prompt.get("relevant_tools", [])
            if isinstance(tool, Mapping)
        }

        if policy == "semantic":
            ranked = sorted(candidates, key=lambda item: item.semantic_score, reverse=True)
            selected_tool = ranked[0].name if ranked else None
            top3 = [candidate.name for candidate in ranked[:3]]
            selected_meta = {
                "score": ranked[0].semantic_score if ranked else 0.0,
                "semantic": ranked[0].semantic_score if ranked else 0.0,
                "functional": ranked[0].semantic_score if ranked else 0.0,
                "query_intent": None,
                "tool_intent": None,
            }
        elif policy == "adaptive":
            result = selector.select(str(prompt.get("query", "")), candidates)
            selected_tool = result.tool_name if result else None
            top3 = [str(row["name"]) for row in result.candidates[:3]] if result else []
            selected_meta = result.candidates[0] if result and result.candidates else {}
        elif policy == "sonar":
            result = sonar_selector.select(str(prompt.get("query", "")), candidates)
            selected_tool = result.tool_name if result else None
            top3 = [str(row["name"]) for row in result.candidates[:3]] if result else []
            selected_meta = result.candidates[0] if result and result.candidates else {}
        else:
            raise ValueError(f"Unknown benchmark policy: {policy}")

        if selected_tool is None:
            return self._empty_row(prompt, policy)

        env.reset(dict(prompt))
        _, reward, _, info = env.step(selected_tool)
        feedback = ToolExecutionFeedback(
            tool_name=selected_tool,
            success=bool(info.get("success", False)),
            latency=float(info.get("latency", 0.0)),
            retries=0,
            semantic_score=float(info.get("semantic_score", 0.0)),
        )
        selector.update(feedback)

        return {
            "policy": policy,
            "query_id": prompt.get("query_id"),
            "query": prompt.get("query"),
            "selected_tool": selected_tool,
            "target_tool": prompt.get("target_tool"),
            "relevant_tools": sorted(name for name in relevant_names if name),
            "target_in_top3": bool(relevant_names.intersection(top3)),
            "is_relevant": selected_tool in relevant_names,
            "success": bool(info.get("success", False)),
            "latency": float(info.get("latency", 0.0)),
            "reward": float(reward),
            "retries": 0,
            "semantic_score": float(selected_meta.get("semantic", 0.0)),
            "functional_score": float(selected_meta.get("functional", 0.0)),
            "selection_score": float(selected_meta.get("score", 0.0)),
            "query_intent": selected_meta.get("query_intent"),
            "tool_intent": selected_meta.get("tool_intent"),
            "rerank_adjustment": float(selected_meta.get("rerank_adjustment", 0.0)),
            "action_overlap": float(selected_meta.get("action_overlap", 0.0)),
            "provider_overlap": float(selected_meta.get("provider_overlap", 0.0)),
            "entity_overlap": float(selected_meta.get("entity_overlap", 0.0)),
            "generic_penalty": float(selected_meta.get("generic_penalty", 0.0)),
            "group_adjustment": float(selected_meta.get("group_adjustment", 0.0)),
            "network_mode": self.network_mode.value,
            "top3": top3,
        }

    def _sample_prompts(self, episodes: int) -> List[Mapping[str, object]]:
        """Handle sample prompts."""
        pool = [p for p in self.config.val_prompts if p.get("relevant_tools")]
        if not pool:
            pool = [p for p in self.config.train_prompts if p.get("relevant_tools")]
        if not pool:
            pool = self.config.prompts
        return random.sample(pool, min(episodes, len(pool)))

    def _empty_row(self, prompt: Mapping[str, object], policy: str) -> Dict[str, object]:
        """Handle empty row."""
        return {
            "policy": policy,
            "query_id": prompt.get("query_id"),
            "query": prompt.get("query"),
            "selected_tool": None,
            "target_tool": prompt.get("target_tool"),
            "relevant_tools": [],
            "target_in_top3": False,
            "is_relevant": False,
            "success": False,
            "latency": 0.0,
            "reward": 0.0,
            "retries": 0,
            "semantic_score": 0.0,
            "functional_score": 0.0,
            "selection_score": 0.0,
            "query_intent": None,
            "tool_intent": None,
            "rerank_adjustment": 0.0,
            "action_overlap": 0.0,
            "provider_overlap": 0.0,
            "entity_overlap": 0.0,
            "generic_penalty": 0.0,
            "group_adjustment": 0.0,
            "network_mode": self.network_mode.value,
            "top3": [],
        }


def print_summary(summary: BenchmarkSummary) -> None:
    """Print print summary."""
    print("\n" + "=" * 60)
    print(f"Benchmark: {summary.policy}")
    print("=" * 60)
    print(f"Episodes:      {summary.episodes}")
    print(f"Relevance@1:   {summary.relevance_at_1:.2%}")
    print(f"Relevance@3:   {summary.relevance_at_3:.2%}")
    print(f"Success rate:  {summary.success_rate:.2%}")
    print(f"Avg latency:   {summary.avg_latency:.3f}s")
    print(f"Avg reward:    {summary.avg_reward:.3f}")
    print(f"Avg retries:   {summary.avg_retries:.3f}")
    if summary.log_path:
        print(f"JSONL log:      {summary.log_path}")


def summary_to_dict(summary: BenchmarkSummary) -> Dict[str, object]:
    """Handle summary to dict."""
    return asdict(summary)
