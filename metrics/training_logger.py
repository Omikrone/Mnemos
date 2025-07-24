import csv
from pyparsing import Path

from config.paths import TRAINING_LOG_PATH


class TrainingLogger:
    """ Logger for training metrics. """

    metrics: list[tuple[int, float, float]]
    log_path: Path

    def __init__(self, log_path: Path = TRAINING_LOG_PATH):
        self.log_path = log_path
        self.metrics = []


    def log(self, epoch: int, step: int, loss: float, perplexity: float):
        """ Log training metrics for a specific epoch. """

        self.metrics.append((epoch, step, loss, perplexity))
        self._save_to_csv()


    def _save_to_csv(self):
        """ Save the logged metrics to a CSV file. """

        if not self.log_path.parent.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss', 'Perplexity'])
            writer.writerows(self.metrics)