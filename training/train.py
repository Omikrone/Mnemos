from pathlib import Path
import pickle
import numpy as np
import json

from model.transformer_model import TransformerModel
from training.batch import BatchBuilder
from training.cross_entropy import CrossEntropyLoss
from training.preprocesser import PreProcesser
from training.tokenizer import BPETokenizer


TRAINING_DATA_PATH = Path("training_data/train_small.txt")
SAVE_MODEL_PATH = Path("save/model.pkl")
SAVE_VOCABULARY_PATH = Path("save/vocabulary.json")
LEARNING_RATE = 1e-3


class Trainer:
    """ Trainer class for training the Transformer model. """

    tokenizer : BPETokenizer
    model : TransformerModel
    loss_fn : CrossEntropyLoss
    lr : int


    def __init__(self):
        """ Initialize the Trainer with the different components. """

        preprocesser = PreProcesser()
        cleaned_data = preprocesser(TRAINING_DATA_PATH)
        self.tokenizer = BPETokenizer(cleaned_data)
        self.tokenizer.build()
        vocab_size = self.tokenizer.get_vocabulary_size()

        self.model = TransformerModel(vocab_size)
        self.lr = LEARNING_RATE
        self.loss_fn = CrossEntropyLoss()


    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """ Perform a single training step and return the loss. """

        # Forward pass
        logits = self.model.forward(input_ids)
        loss = self.loss_fn(logits, targets)

        # Backward pass
        loss_gradient = self.loss_fn.backward(logits, targets)
        self.model.backward(loss_gradient)
        self.model.step(self.lr)
        self.model.zero_grad()

        return loss
    

    def train(self):
        """ Run the training loop and save the model after training. """

        print("Starting training...")

        # Perform the chunking and create batches from the training data
        batch_builder = BatchBuilder(self.tokenizer.text, self.tokenizer)
        chunks = batch_builder.create_chunks()
        batches = batch_builder.create_batches(chunks)
        total_batches = len(batches)

        print(f"Number of batches: {total_batches}")

        # Training loop with 5 epochs
        for epoch in range(1, 6):
            print(f"Epoch {epoch}/{5}")
            np.random.shuffle(batches)

            for i, batch in enumerate(batches):
                loss = self.train_step(batch[0], batch[1])

                # Display progress every 1% of the total batches
                if i % max(1, total_batches // 100) == 0:
                    percent = (i / total_batches) * 100
                    print(f"[{i}/{total_batches}] {percent:.1f}% - Loss: {loss:.4f}")

        self.save_model()
        print("Model saved successfully.")


    def save_model(self):
        """ Save the model parameters and vocabulary to files. """
        
        model_parameters = self.model.get_parameters()
        with open(SAVE_MODEL_PATH, "wb") as f:
            pickle.dump(model_parameters, f)

        vocabulary = self.tokenizer.table_manager.load_table()
        with open(SAVE_VOCABULARY_PATH, "w") as f:
            json.dump(vocabulary, f, indent=4)