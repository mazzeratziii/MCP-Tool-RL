"""
python -m src.rl.eval_gap

Diagnoses train/val gap:
- Runs greedy eval on 200 TRAIN prompts (memorisation check)
- Runs greedy eval on 200 VAL prompts (generalisation)
- Compares coverage between train and val sets
- Checks if val queries are harder (lower semantic scores)
"""
import sys, os, random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode

TOP_K = 20
N_EVAL = 200


def check_coverage(prompts, env, label, n=200):
    sample = random.sample(prompts, min(n, len(prompts)))
    hits = sem_rel, sem_all = 0, [], []
    hits = 0
    sem_rel, sem_all = [], []

    for prompt in sample:
        state = env.reset(prompt)
        candidates = {t["name"] for t in state["tools"]}
        relevant = {t["name"] for t in prompt.get("relevant_tools", [])}

        if candidates & relevant:
            hits += 1

        for t in state["tools"]:
            s = t.get("semantic_score", 0.5)
            sem_all.append(s)
            if t["name"] in relevant:
                sem_rel.append(s)

    cov = hits / len(sample)
    avg_sem_rel  = sum(sem_rel) / len(sem_rel) if sem_rel else 0
    avg_sem_all  = sum(sem_all) / len(sem_all) if sem_all else 0
    print(f"\n{label}:")
    print(f"  Coverage (relevant in top-{TOP_K}): {cov:.1%}  ({hits}/{len(sample)})")
    print(f"  Avg semantic score — relevant: {avg_sem_rel:.4f}  all: {avg_sem_all:.4f}")
    print(f"  Gap (relevant - all): {avg_sem_rel - avg_sem_all:.4f}")
    return cov


def main():
    config = Config()
    config.load_data()
    config.profile = "desktop" if (
        torch.cuda.is_available() and
        torch.cuda.get_device_properties(0).total_memory // 1024**2 >= 7000
    ) else "laptop"

    # Use the override top_k
    from src.environment.tool_registry import ToolRegistry

    env = MCPEnvironment(config, llm_client=None,
                         network_mode=NetworkMode.DETERMINISTIC)

    # Override get_top_k in env to use TOP_K
    orig_reset = env.reset
    def reset_with_topk(prompt=None):
        state = orig_reset(prompt)
        candidates = env.tools.get_top_k_tools(env.current_query, k=TOP_K)
        tools_state = []
        for tool in candidates:
            sv  = env.network.get_server_state(tool["name"])
            qos = env.network.get_qos_metrics(tool["name"])
            sem = env.tools.semantic_similarity(env.current_query, tool["name"])
            is_rel = any(rt["name"] == tool["name"] for rt in env.relevant_tools)
            tools_state.append({
                "name": tool["name"],
                "semantic_score": sem,
                "is_relevant": is_rel,
                "available": sv["available"],
                "latency": qos["avg_latency"],
                "stability": qos["stability"],
                "category": tool.get("category", ""),
                "description": tool.get("description", "")[:50] + "...",
                "used": False,
            })
        state["tools"] = tools_state
        return state
    env.reset = reset_with_topk

    train_pool = [p for p in config.train_prompts if p.get("relevant_tools")]
    val_pool   = [p for p in config.val_prompts   if p.get("relevant_tools")]

    print(f"Train pool: {len(train_pool)}  Val pool: {len(val_pool)}")

    cov_train = check_coverage(train_pool, env, f"TRAIN (top-{TOP_K})", N_EVAL)
    cov_val   = check_coverage(val_pool,   env, f"VAL   (top-{TOP_K})", N_EVAL)

    print(f"\n=== SUMMARY ===")
    print(f"  Coverage gap (train - val): {cov_train - cov_val:.1%}")
    if cov_train - cov_val > 0.1:
        print("  ⚠ Large gap → val queries are harder for the retriever")
        print("    Fix: increase TOP_K_TOOLS further, or tune sentence-transformer")
    else:
        print("  ✓ Coverage similar → gap is model overfitting, not retriever issue")
        print("    Fix: more regularisation (higher entropy_coeff, lower lora_r)")


if __name__ == "__main__":
    main()