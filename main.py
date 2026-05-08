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
    parser = argparse.ArgumentParser(description="NetMCP RL Training")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "interactive"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--network-mode", type=str, default="deterministic",
                        choices=["deterministic", "controlled", "stochastic"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = Config()
    config.load_data()

    config.rl.batch_size = max(1, args.batch_size)
    config.rl.num_epochs = args.epochs

    network_mode_map = {
        "deterministic": NetworkMode.DETERMINISTIC,
        "controlled":    NetworkMode.CONTROLLED,
        "stochastic":    NetworkMode.STOCHASTIC,
    }
    network_mode = network_mode_map[args.network_mode]

    print(f"Mode={args.mode}  batch_size={config.rl.batch_size}  "
          f"epochs={config.rl.num_epochs}  network={args.network_mode}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    trainer = NetMCPTrainer(config)

    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            trainer.load_checkpoint(args.checkpoint)
        except Exception as e:
            print(f"Checkpoint load error: {e}")

    if args.mode == "train":
        trainer.train()
    elif args.mode == "evaluate":
        trainer.evaluate(network_mode=network_mode)
    elif args.mode == "interactive":
        run_interactive(trainer)


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def run_interactive(trainer: "NetMCPTrainer"):
    print("\n" + "=" * 60)
    print("NetMCP Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /network deterministic | controlled | stochastic")
    print("  /network stats")
    print("  /help   /exit")
    print("=" * 60)
    print(f"Loaded {len(trainer.config.tools)} tools\n")

    network_mode_map = {
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

        if query.lower() in ("quit", "exit", "/exit"):
            break

        # ---- Commands ----
        if query.startswith("/"):
            parts = query[1:].split()
            cmd = parts[0].lower() if parts else ""

            if cmd == "help":
                print("\n/network deterministic | controlled | stochastic — switch mode")
                print("/network stats — show current stats")
                print("/exit — quit\n")

            elif cmd == "network" and len(parts) > 1:
                sub = parts[1].lower()
                if sub == "stats":
                    stats = trainer.env.get_network_stats()
                    for k, v in stats.items():
                        print(f"  {k}: {v}")
                    print()
                elif sub in network_mode_map:
                    trainer.env.set_network_mode(network_mode_map[sub])
                    print(f"Network mode → {sub}")
                else:
                    print(f"Unknown sub-command: {sub}")
            else:
                print(f"Unknown command: {cmd}. Type /help.")
            continue

        # ---- Query ----
        query_data = {"query": query, "domain": "user_query", "relevant_tools": []}
        state = trainer.env.reset(query_data)

        print(f"\nProcessing: {query}")
        print("-" * 50)

        available = [t["name"] for t in state["tools"] if t.get("available", True)]
        if not available:
            print("No available tools for this query.")
            print("-" * 50)
            continue

        resolved = False
        import torch

        with torch.no_grad():
            for step in range(trainer.config.rl.max_steps):
                context = trainer._format_context(state)
                tools = [t["name"] for t in state["tools"]]

                logits, _ = trainer._forward(context, tools)

                # Greedy decode in interactive mode
                action_idx = logits.argmax().item()
                tool_name = tools[action_idx]

                if tool_name not in available:
                    print(f"  Step {step+1}: {tool_name} not available, skipping")
                    continue

                next_state, _, done, info = trainer.env.step(tool_name)

                if info.get("success"):
                    response = info.get("response") or info.get("result", "")
                    print(f"\nRESPONSE:")
                    print(response or f"Request handled by '{tool_name}'")
                    print(f"\n  Tool:    {tool_name}")
                    print(f"  Latency: {info.get('latency', 0):.3f}s")
                    resolved = True
                    break
                else:
                    print(f"  Step {step+1}: {tool_name} failed — trying next")
                    state = next_state
                    if done:
                        break

        if not resolved:
            print("\nCould not resolve query with available tools.")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()