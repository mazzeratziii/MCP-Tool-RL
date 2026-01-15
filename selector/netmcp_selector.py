class NetMCPSelector:
    def __init__(self, metrics_store):
        self.metrics = metrics_store

    def select(self, tool_names, query):
        best_tool = None
        best_score = float("-inf")

        for tool in tool_names:
            stats = self.metrics.get_stats(tool)

            # если данных нет — exploration
            if stats is None:
                return tool

            score = (
                + stats["success_rate"] * 1.0
                - stats["avg_latency"] * 0.5
                - stats["jitter"] * 0.2
            )

            if score > best_score:
                best_score = score
                best_tool = tool

        return best_tool
