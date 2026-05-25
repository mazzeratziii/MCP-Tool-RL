import csv
import os
import tempfile
import unittest

from src.rl.training_plot import TrainingMetric, TrainingProgressLogger


class TrainingProgressLoggerTest(unittest.TestCase):
    def test_writes_training_metrics_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "metrics.csv")
            plot_path = os.path.join(tmp, "curve.png")
            logger = TrainingProgressLogger(csv_path=csv_path, plot_path=plot_path)

            logger.append(
                TrainingMetric(
                    epoch=1,
                    loss=0.5,
                    reward=2.0,
                    success_rate=0.9,
                    relevance_rate=0.7,
                    updates=10,
                    skipped=1,
                    rollouts=30,
                    adv_std=0.2,
                    lr=1e-5,
                    scenario="stable",
                )
            )

            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(rows[0]["epoch"], "1")
            self.assertEqual(rows[0]["scenario"], "stable")


if __name__ == "__main__":
    unittest.main()
