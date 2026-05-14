# main.py
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.config import Config
from src.rl.train_grpo import NetMCPTrainer
from src.environment.network_emulator import NetworkMode


def main():
    parser = argparse.ArgumentParser(description="NetMCP-RL")
    parser.add_argument("--mode", default="train",
                        choices=["train", "evaluate", "interactive"])
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--network-mode", default="deterministic",
                        choices=["deterministic", "controlled", "stochastic"])
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--profile",      default=None,
                        choices=["desktop", "laptop"],
                        help="Hardware profile. Auto-detected if omitted.")
    parser.add_argument("--eval-episodes", type=int, default=200,
                        help="Number of episodes for evaluate mode")
    parser.add_argument("--use-hybrid", action="store_true",
                        help="Use hybrid mode: real MCP calls + emulation")
    parser.add_argument("--mcp-config", default="mcp_config.json",
                        help="Path to MCP configuration file")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = Config()
    config.load_data()
    config.rl.batch_size  = max(1, args.batch_size)
    config.rl.num_epochs  = args.epochs

    # ── Profile auto-detection ────────────────────────────────────────
    if args.profile:
        config.profile = args.profile
    elif torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024 ** 2
        config.profile = "desktop" if vram_mb >= 7000 else "laptop"
        print(f"Auto-detected profile: {config.profile} ({vram_mb} MB VRAM)")
    else:
        config.profile = "laptop"

    network_mode_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled":    NetworkMode.CONTROLLED,
        "stochastic":    NetworkMode.STOCHASTIC,
    }
    network_mode = network_mode_map[args.network_mode]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # ── Выбор режима: гибридный или эмуляция ──────────────────────────
    if args.use_hybrid:
        print("\n" + "=" * 60)
        print("HYBRID MODE: Real MCP + Emulation")
        print("=" * 60)
        from src.rl.train_grpo_hybrid import HybridMCPTrainer
        trainer = HybridMCPTrainer(config, mcp_config_path=args.mcp_config)
    else:
        print("\n" + "=" * 60)
        print("EMULATION MODE: All tools emulated")
        print("=" * 60)
        trainer = NetMCPTrainer(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Checkpoint load error: {e}")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate(num_episodes=args.eval_episodes, network_mode=network_mode)
    elif args.mode == "interactive":
        run_interactive(trainer)


# ── Interactive mode ──────────────────────────────────────────────────────────

def run_interactive(trainer: "NetMCPTrainer"):
    print("\n" + "=" * 60)
    print("NetMCP Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /network deterministic | controlled | stochastic")
    print("  /network stats")
    print("  /eval N   — quick eval on N val prompts")
    print("  /help   /exit")
    print("=" * 60)
    print(f"Profile: {trainer.p.name}")
    print(f"Tools:   {len(trainer.config.tools)}\n")

    nm_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled":    NetworkMode.CONTROLLED,
        "stochastic":    NetworkMode.STOCHASTIC,
    }

    trainer.model.eval()
    while True:
        try:
            query = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit", "/exit"):
            break

        # Commands
        if query.startswith("/"):
            parts = query[1:].split()
            cmd   = parts[0].lower() if parts else ""

            if cmd == "help":
                print("  /network [deterministic|controlled|stochastic|stats]")
                print("  /eval N   — evaluate on N val prompts")
                print("  /exit")

            elif cmd == "network" and len(parts) > 1:
                sub = parts[1].lower()
                if sub == "stats":
                    s = trainer.env.get_network_stats()
                    for k, v in s.items():
                        print(f"  {k}: {v}")
                elif sub in nm_map:
                    trainer.env.set_network_mode(nm_map[sub])
                    print(f"Network → {sub}")
                else:
                    print(f"Unknown: {sub}")

            elif cmd == "eval":
                n = int(parts[1]) if len(parts) > 1 else 50
                trainer.evaluate(num_episodes=n,
                                 network_mode=trainer.env.network.mode)

            else:
                print(f"Unknown command: {cmd}")
            continue

        # Regular query
        query_data = {"query": query, "domain": "user_query", "relevant_tools": []}
        state, tools, scores = trainer._get_tools_state(query_data)
        print(f"\nTop tools for: '{query}'")
        print("-" * 50)

        import torch
        with torch.inference_mode():
            ids    = trainer._encode_context(state)
            embs   = trainer._build_tool_embs(tools)
            logits = trainer._forward_with_embs(ids, embs, scores)
            probs  = torch.softmax(logits, dim=-1)

        # Show top-5
        top5 = probs.topk(min(5, len(tools)))
        for rank, (prob, idx) in enumerate(
            zip(top5.values.tolist(), top5.indices.tolist()), 1
        ):
            tool = tools[idx]
            sem  = scores[idx]
            print(f"  {rank}. {tool}")
            print(f"     model_prob={prob:.3f}  semantic={sem:.3f}")

        # Execute top-1
        best_tool = tools[logits.argmax().item()]
        trainer.env.reset(query_data)
        _, _, _, info = trainer.env.step(best_tool)
        print(f"\nExecuted: {best_tool}")
        print(f"Success:  {info.get('success', False)}")
        print(f"Latency:  {info.get('latency', 0):.3f}s")
        resp = info.get("response") or info.get("result", "")
        if resp:
            print(f"Response: {resp[:200]}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()