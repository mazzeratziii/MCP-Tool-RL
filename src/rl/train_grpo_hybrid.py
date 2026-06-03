"""
Hybrid MCP Trainer: Uses real MCP calls for some tools, emulation for others
"""
from src.rl.train_grpo import NetMCPTrainer
from src.environment.hybrid_environment import HybridMCPEnvironment
from src.environment.network_emulator import NetworkMode


class HybridMCPTrainer(NetMCPTrainer):
    """
    Trainer с гибридным окружением:
    - Реальные MCP вызовы для зарегистрированных инструментов
    - Эмуляция для остальных
    """

    def __init__(self, config, mcp_config_path: str = "mcp_config.json"):
        # Инициализируем базовый trainer
        """Initialize the object."""
        super().__init__(config)

        # Заменяем окружение на гибридное
        self.env = HybridMCPEnvironment(
            config,
            mcp_config_path=mcp_config_path
        )

        print(f"\nHybrid environment initialized:")
        print(f"  MCP tools: {len(self.env.mcp_tools)}")
        print(f"  Emulated tools: {len(config.tools) - len(self.env.mcp_tools)}")

    def __del__(self):
        """Деструктор для закрытия MCP клиента"""
        if hasattr(self, 'env'):
            self.env.close()
