"""
python -m src.rl.diagnose

Runs without training — checks:
1. What fraction of top-10 candidates actually contain a relevant tool
2. Whether the model already prefers relevant tools (pre-training bias)
3. Reward distribution
"""
import random
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import torch
import torch.nn.functional as F

from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode
from src.prompts import get_dynamic_prompt


def main():
    config = Config()
    config.load_data()

    env = MCPEnvironment(config, llm_client=None, network_mode=NetworkMode.DETERMINISTIC)

    pool = [p for p in config.prompts if p.get("relevant_tools")]
    sample = random.sample(pool, min(200, len(pool)))

    # ── 1. Coverage: does top-10 contain the relevant tool? ──────────────
    coverage_hits = 0
    for prompt in sample:
        state = env.reset(prompt)
        candidate_names = {t["name"] for t in state["tools"]}
        relevant_names = {t["name"] for t in prompt.get("relevant_tools", [])}
        if candidate_names & relevant_names:
            coverage_hits += 1

    print(f"\n=== COVERAGE (top-10 contains relevant tool) ===")
    print(f"  {coverage_hits}/{len(sample)} = {coverage_hits/len(sample):.1%}")
    print(f"  → If this is < 50%, increase k in get_top_k_tools()")

    # ── 2. Reward distribution ────────────────────────────────────────────
    from src.rl.reward_functions import GRPOToolReward
    rf = GRPOToolReward(config)

    rewards_relevant = []
    rewards_random = []

    for prompt in sample[:100]:
        state = env.reset(prompt)
        tools = [t["name"] for t in state["tools"]]
        relevant_names = {t["name"] for t in prompt.get("relevant_tools", [])}

        for tool_name in tools:
            env.reset(prompt)
            _, _, _, info = env.step(tool_name)
            r = rf.compute_outcome_reward(
                success=info.get("success", False),
                steps=1,
                is_relevant=info.get("is_relevant", False),
                latency=info.get("latency", 0.0),
                semantic_score=info.get("semantic_score", 0.0),
            )
            if tool_name in relevant_names:
                rewards_relevant.append(r)
            else:
                rewards_random.append(r)

    def stats(lst, label):
        if not lst:
            print(f"  {label}: no data")
            return
        print(f"  {label}: mean={sum(lst)/len(lst):.3f}  "
              f"min={min(lst):.3f}  max={max(lst):.3f}  n={len(lst)}")

    print(f"\n=== REWARD DISTRIBUTION ===")
    stats(rewards_relevant, "relevant tools")
    stats(rewards_random,   "non-relevant  ")
    gap = (sum(rewards_relevant)/len(rewards_relevant) if rewards_relevant else 0) - \
          (sum(rewards_random)/len(rewards_random) if rewards_random else 0)
    print(f"  gap (relevant - random): {gap:.3f}")
    print(f"  → Gap should be > 1.0 for a strong RL signal")

    # ── 3. Semantic score of relevant vs random tools ────────────────────
    from src.environment.tool_registry import ToolRegistry
    registry = ToolRegistry(config)

    sem_rel, sem_rand = [], []
    for prompt in sample[:50]:
        state = env.reset(prompt)
        tools_in_state = [t["name"] for t in state["tools"]]
        relevant_names = {t["name"] for t in prompt.get("relevant_tools", [])}
        for name in tools_in_state:
            score = registry.semantic_similarity(prompt["query"], name)
            if name in relevant_names:
                sem_rel.append(score)
            else:
                sem_rand.append(score)

    print(f"\n=== SEMANTIC SIMILARITY (ToolRegistry top-10) ===")
    stats(sem_rel,  "relevant tools")
    stats(sem_rand, "non-relevant  ")

    print("\n=== SUMMARY ===")
    if coverage_hits / len(sample) < 0.5:
        print("  ❌ Coverage < 50% — relevant tools rarely appear in top-10.")
        print("     Fix: increase k in get_top_k_tools() to 20 or 30.")
    else:
        print("  ✓ Coverage OK")

    if gap < 1.0:
        print("  ❌ Reward gap < 1.0 — model can't distinguish relevant from random.")
        print("     Fix: increase success_reward or relevance bonus in reward_functions.py")
    else:
        print("  ✓ Reward gap OK")


if __name__ == "__main__":
    main()