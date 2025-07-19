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
    """ Tester class for evaluating the model. """

    tokenizer: BPETokenizer
    model: TransformerModel
    loss_fn: CrossEntropyLoss

    def __init__(self, model_path: Path = Path("save/model.pkl"), vocab_path: Path = Path("save/vocabulary.json")):
        """ Initialize the Tester with the model and tokenizer. """

        # Load the different components
        preprocesser = PreProcesser()
        cleaned_data = preprocesser(TEST_DATA_FILE)
        self.tokenizer = BPETokenizer(cleaned_data)
        self.model = self.load_model(model_path, vocab_path)
        self.loss_fn = CrossEntropyLoss()
    

    def load_model(self, model_path: Path) -> TransformerModel:
        """ Load a saved model from a file. """

        with open(model_path, "rb") as f:
            model_params = pickle.load(f)

        return TransformerModel.from_params(model_params)
    

    def test_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """ Perform a single test step and return the loss. """

        logits = self.model.forward(input_ids)
        loss = self.loss_fn(logits, targets)
        return loss
    
    
    def test(self):
        """ Run the test and print the average loss. """

        print("Starting testing...")

        # Perform the chunking and create batches from the test data
        batch_builder = BatchBuilder(self.tokenizer.text, self.tokenizer)
        chunks = batch_builder.create_chunks()
        batches = batch_builder.create_batches(chunks)
        global_loss = 0.0

        for batch in batches:
            loss = self.test_step(batch[0], batch[1])
            global_loss += loss

        print("Global Loss : " + str(global_loss / len(batches)))