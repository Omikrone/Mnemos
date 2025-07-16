from pathlib import Path
import pickle
import numpy as np
import json

from training.tokenizer import Tokenizer
from model.transformer_model import TransformerModel
from training.cross_entropy import CrossEntropyLoss
from training.preprocesser import PreProcesser
from training.bpe import bpe_builder, get_vocabulary_size, encode_text


TRAINING_DATA_PATH = Path("training_data/train_small.txt")
SAVE_MODEL_PATH = Path("save/model.pkl")
SAVE_VOCABULARY_PATH = Path("save/vocabulary.json")


class Trainer:

    tokenizer : Tokenizer
    model : TransformerModel
    loss_fn : CrossEntropyLoss
    lr : int

    def __init__(self):
        preprocesser = PreProcesser()
        cleaned_data = preprocesser(TRAINING_DATA_PATH)
        self.tokenizer = Tokenizer(cleaned_data)
        bpe_builder(cleaned_data)
        encode_text(cleaned_data)
        vocab_size = get_vocabulary_size()

        self.model = TransformerModel(vocab_size)
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

        print("Starting training...")
        chunks = self.tokenizer.create_chunks()
        batches = self.tokenizer.create_batches(chunks)
        total_batches = len(batches)
        print(f"Number of batches: {total_batches}")

        for i, batch in enumerate(batches):
            loss = self.train_step(batch[0], batch[1])

            # Affichage tous les 1 %
            if i % max(1, total_batches // 100) == 0:
                percent = (i / total_batches) * 100
                print(f"[{i}/{total_batches}] {percent:.1f}% - Loss: {loss:.4f}")

        self.save_model()
        print("Model saved successfully.")


    def save_model(self):
        model_parameters = self.model.get_parameters()
        with open(SAVE_MODEL_PATH, "wb") as f:
            pickle.dump(model_parameters, f)

        vocabulary = self.tokenizer.table_manager.load_table()
        with open(SAVE_VOCABULARY_PATH, "w") as f:
            json.dump(vocabulary, f, indent=4)