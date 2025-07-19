import json
from pathlib import Path
import pickle

import numpy as np

from model.transformer_model import TransformerModel
from training.batch import BatchBuilder
from training.preprocesser import PreProcesser
from training.cross_entropy import CrossEntropyLoss
from training.tokenizer import BPETokenizer


TEST_DATA_FILE = Path("test_data/test1.txt")



class Tester:

    tokenizer: BPETokenizer
    model: TransformerModel
    loss_fn: CrossEntropyLoss

    def __init__(self, model_path: Path = Path("save/model.pkl"), vocab_path: Path = Path("save/vocabulary.json")):
        preprocesser = PreProcesser()
        cleaned_data = preprocesser(TEST_DATA_FILE)
        self.tokenizer = BPETokenizer(cleaned_data)
        self.model = self.load_model(model_path, vocab_path)
        self.loss_fn = CrossEntropyLoss()
    

    def load_model(self, model_path: Path, vocab_path: Path) -> TransformerModel:
        """ Load the model from a file. """

        with open(model_path, "rb") as f:
            model_params = pickle.load(f)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        return TransformerModel.from_params(model_params)
    

    def test_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        logits = self.model.forward(input_ids)
        loss = self.loss_fn(logits, targets)
        return loss
    
    
    def test(self):
        print("Starting testing...")
        batch_builder = BatchBuilder(self.tokenizer.text, self.tokenizer)
        chunks = batch_builder.create_chunks()
        batches = batch_builder.create_batches(chunks)
        global_loss = 0.0

        for batch in batches:
            loss = self.test_step(batch[0], batch[1])
            global_loss += loss

        print("Global Loss : " + str(global_loss / len(batches)))