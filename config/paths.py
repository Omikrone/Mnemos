from pathlib import Path

ROOT_DIR = Path('.')


SAVED_MODEL_DIR = ROOT_DIR / "save"
MODEL_PATH = SAVED_MODEL_DIR / "model.pkl"
VOCABULARY_PATH = SAVED_MODEL_DIR / "table.json"
MERGES_PATH = SAVED_MODEL_DIR / "merges.pkl"


DATA_DIR = ROOT_DIR / 'data'

TRAINING_DATA_DIR = DATA_DIR / "training"
TRAINING_DATA_PATH = TRAINING_DATA_DIR / "medium_train.txt"

TEST_DATA_DIR = DATA_DIR / "testing"
TEST_DATA_PATH = TEST_DATA_DIR / "test.txt"


LOGS_DIR = ROOT_DIR / "logs"
TRAINING_LOG_PATH = LOGS_DIR / "training_log.csv"
VALIDATION_LOG_PATH = LOGS_DIR / "validation_log.csv"