import csv
from pathlib import Path

from mnemos.config.paths import TRAINING_LOG_PATH, VALIDATION_LOG_PATH


class TrainingLogger:
    """ Logger for training metrics. """

    training_metrics: list[tuple[int, float, float]]
    validation_metrics: list[tuple[int, float, float]]
    log_path: Path

    def __init__(self, log_path: Path = TRAINING_LOG_PATH, validation_log_path: Path = VALIDATION_LOG_PATH):
        """ Initialize the logger with the path to save logs. """
        self.training_log_path = log_path
        self.validation_log_path = validation_log_path
        self.training_metrics = []
        self.validation_metrics = []


    def train_log(self, epoch: int, step: int, loss: float, perplexity: float):
        """ Log training metrics for a specific epoch. """

        self.training_metrics.append((epoch, step, loss, perplexity))
        self._save_to_csv()
    

    def validation_log(self, epoch: int, step: int, loss: float, perplexity: float):
        """ Log validation metrics for a specific epoch. """

        self.validation_metrics.append((epoch, step, loss, perplexity))
        self._save_to_csv()


    def _save_to_csv(self):
        """ Save the logged metrics to a CSV file. """

        if not self.training_log_path.parent.exists():
            self.training_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.training_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss', 'Perplexity'])
            writer.writerows(self.training_metrics)

        with open(self.validation_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Step', 'Loss', 'Perplexity'])
            writer.writerows(self.validation_metrics)