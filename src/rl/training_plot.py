import csv
import os
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional


@dataclass
class TrainingMetric:
    epoch: int
    loss: float
    reward: float
    success_rate: float
    relevance_rate: float
    updates: int
    skipped: int
    rollouts: int
    adv_std: float
    lr: float
    scenario: str = ""


class TrainingProgressLogger:
    """Сохраняет метрики обучения и строит компактный график прогресса."""

    def __init__(
        self,
        csv_path: str = "runs/training_metrics.csv",
        plot_path: str = "runs/training_curve.png",
        enabled: bool = True,
    ):
        """?????????????? ?????? ? ????????? ??????????? ???????????."""
        self.csv_path = csv_path
        self.plot_path = plot_path
        self.enabled = enabled
        self.metrics: List[TrainingMetric] = []
        self._plot_warning_shown = False

        if self.enabled:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(self.plot_path) or ".", exist_ok=True)

    def append(self, metric: TrainingMetric) -> None:
        """????????? ??????? ????? ? ????????? ????? ?????????."""
        if not self.enabled:
            return
        self.metrics.append(metric)
        self._write_csv()
        try:
            self.render_plot()
        except RuntimeError as exc:
            if not self._plot_warning_shown:
                print(f"График обучения пропущен: {exc}")
                self._plot_warning_shown = True

    def _write_csv(self) -> None:
        """?????????? ??????? ?????? ???????? ? CSV."""
        fieldnames = list(asdict(self.metrics[0]).keys()) if self.metrics else []
        if not fieldnames:
            return
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(asdict(metric))

    def render_plot(self) -> Optional[str]:
        """?????? ?????? ?? ??????? ???????? ????????."""
        return render_training_plot(self.metrics, self.plot_path)


def render_training_plot(
    metrics: Iterable[TrainingMetric],
    output_path: str = "runs/training_curve.png",
) -> Optional[str]:
    """?????? PNG-?????? ???????? ????????."""
    metrics = list(metrics)
    if not metrics:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib недоступен ({exc})") from exc

    epochs = [m.epoch for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    fig.suptitle("Прогресс GRPO-обучения", fontsize=14, fontweight="bold")

    ax = axes[0][0]
    ax.plot(epochs, [m.reward for m in metrics], color="#2563eb", marker="o", linewidth=2)
    ax.set_title("Средняя награда")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Награда")
    ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(epochs, [m.loss for m in metrics], color="#dc2626", marker="o", linewidth=2)
    ax.set_title("Прокси-loss")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1][0]
    ax.plot(epochs, [m.relevance_rate * 100 for m in metrics], label="Релевантность", color="#16a34a", marker="o", linewidth=2)
    ax.plot(epochs, [m.success_rate * 100 for m in metrics], label="Успех", color="#9333ea", marker="o", linewidth=2)
    ax.set_title("Качество выбора")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Процент")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1][1]
    ax.plot(epochs, [m.updates for m in metrics], label="Обновления", color="#0891b2", marker="o", linewidth=2)
    ax.plot(epochs, [m.skipped for m in metrics], label="Пропущено", color="#f59e0b", marker="o", linewidth=2)
    ax.set_title("Шаги обучения")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Количество")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
