from pathlib import Path
import numpy as np

from training.tokenizer import Tokenizer
from model.transformer_model import TransformerModel
from training.cross_entropy import CrossEntropyLoss
from training.preprocesser import PreProcesser


class Trainer:

    model : TransformerModel
    loss_fn : CrossEntropyLoss
    lr : int

    def __init__(self):
        self.model = TransformerModel(85)
        self.lr = 1e-3
        self.loss_fn = CrossEntropyLoss()

    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        logits = self.model.forward(input_ids)
        loss = self.loss_fn(logits, targets)

        loss_gradient = self.loss_fn.backward(logits, targets)
        self.model.backward(loss_gradient)
        self.model.step(self.lr)
        self.model.zero_grad()

        return loss
    
    def train(self):
        preprocesser = PreProcesser()
        cleaned_data = preprocesser(Path("data/assemblee_nationale.txt"))
        tokenizer = Tokenizer(cleaned_data)

        chunks = tokenizer.create_chunks()
        batches = tokenizer.create_batches(chunks)

        for batch in batches:
            loss = self.train_step(batch[0], batch[1])
            print("Loss : " + str(loss))

