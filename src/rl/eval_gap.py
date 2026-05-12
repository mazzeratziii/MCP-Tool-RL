import sys, os, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)
import torch
from src.config import Config
from src.environment.mcp_environment import MCPEnvironment
from src.environment.network_emulator import NetworkMode

TOP_K  = 20
N_EVAL = 200

def check_coverage(prompts, reset_fn, label, n=200):
    sample = random.sample(prompts, min(n, len(prompts)))
    hits = 0
    sem_rel, sem_all = [], []
    for prompt in sample:
        state = reset_fn(prompt)
        candidates = {t["name"] for t in state["tools"]}
        relevant   = {t["name"] for t in prompt.get("relevant_tools", [])}
        if candidates & relevant:
            hits += 1
        for t in state["tools"]:
            s = t.get("semantic_score", 0.5)
            sem_all.append(s)
            if t["name"] in relevant:
                sem_rel.append(s)
    cov = hits / len(sample)
    avg_rel = sum(sem_rel)/len(sem_rel) if sem_rel else 0
    avg_all = sum(sem_all)/len(sem_all) if sem_all else 0
    print(f"\n{label}:")
    print(f"  Coverage: {cov:.1%} ({hits}/{len(sample)})")
    print(f"  Semantic — relevant: {avg_rel:.4f}  all: {avg_all:.4f}  gap: {avg_rel-avg_all:+.4f}")
    return cov

def main():
    config = Config()
    config.load_data()
    config.profile = "laptop"
    env = MCPEnvironment(config, llm_client=None, network_mode=NetworkMode.DETERMINISTIC)
    def reset_topk(prompt):
        env.reset(prompt)
        candidates = env.tools.get_top_k_tools(env.current_query, k=TOP_K)
        tools_state = []
        for tool in candidates:
            sv  = env.network.get_server_state(tool["name"])
            qos = env.network.get_qos_metrics(tool["name"])
            sem = env.tools.semantic_similarity(env.current_query, tool["name"])
            is_rel = any(rt["name"] == tool["name"] for rt in env.relevant_tools)
            tools_state.append({"name": tool["name"], "semantic_score": sem,
                "is_relevant": is_rel, "available": sv["available"],
                "latency": qos["avg_latency"], "stability": qos["stability"],
                "category": tool.get("category",""), "description": tool.get("description","")[:50]+"...", "used": False})
        return {"tools": tools_state, "query": env.current_query}
    train_pool = [p for p in config.train_prompts if p.get("relevant_tools")]
    val_pool   = [p for p in config.val_prompts   if p.get("relevant_tools")]
    print(f"\nTrain pool: {len(train_pool)}  Val pool: {len(val_pool)}")
    cov_train = check_coverage(train_pool, reset_topk, f"TRAIN (top-{TOP_K})")
    cov_val   = check_coverage(val_pool,   reset_topk, f"VAL   (top-{TOP_K})")
    diff = cov_train - cov_val
    print(f"\n{'='*50}")
    print(f"Coverage gap (train - val): {diff:+.1%}")
    if diff > 0.10:
        print("-> Retriever bias: increase TOP_K to 25-30")
    elif diff > 0.05:
        print("-> Small bias + overfitting. TOP_K 20->25, entropy 0.005->0.01")
    else:
        print("-> Coverage similar. Pure overfitting. Raise entropy_coeff.")

if __name__ == "__main__":
    main()