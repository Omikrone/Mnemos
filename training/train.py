import pickle
import sys
import numpy as np
import json
import csv

from metrics.training_logger import TrainingLogger
from model.transformer_model import TransformerModel
from training.test import Tester
from training.tokenizer.parallel_encoding import tokenize_text
from training.batch import BatchBuilder
from training.cross_entropy import CrossEntropyLoss
from training.preprocesser import PreProcesser
from training.tokenizer.bpe import BPETokenizer
from config.paths import TRAINING_DATA_PATH, MODEL_PATH, VOCABULARY_PATH
from config.params import LEARNING_RATE, NB_EPOCHS


class Trainer:
    """ Trainer class for training the Transformer model. """

    tokenizer : BPETokenizer
    model : TransformerModel
    loss_fn : CrossEntropyLoss
    lr : int
    logger : TrainingLogger


    def __init__(self):
        """ Initialize the Trainer with the different components. """

        preprocesser = PreProcesser()
        cleaned_data = preprocesser(TRAINING_DATA_PATH)
        self.tokenizer = BPETokenizer(cleaned_data)
        self.tokenizer.build()
        vocab_size = self.tokenizer.get_vocabulary_size()

        self.model = TransformerModel(vocab_size)
        #self.model = Tester.load_model(MODEL_PATH)
        self.lr = LEARNING_RATE
        self.loss_fn = CrossEntropyLoss()
        self.logger = TrainingLogger()


    def train_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """ Perform a single training step and return the loss. """

        # Forward pass
        logits = self.model.forward(input_ids, train=True)
        loss = self.loss_fn(logits, targets)

        # Backward pass
        loss_gradient = self.loss_fn.backward(logits, targets)
        self.model.backward(loss_gradient)
        self.model.step(self.lr)
        self.model.zero_grad()

        return loss
    

    def validation_step(self, input_ids: np.ndarray, targets: np.ndarray) -> float:
        """ Perform a single validation step and return the loss. """

        # Forward pass
        logits = self.model.forward(input_ids, train=False)
        loss = self.loss_fn(logits, targets)

        return loss
    

    def train(self):
        """ Run the training loop and save the model after training. """

        print("\nStarting training...")
        print("Creating batches for training...")

        if not TRAINING_DATA_PATH.exists():
            print(f"Training data file not found: {TRAINING_DATA_PATH}")
            return

        # Perform the chunking and create batches from the training data
        batch_builder = BatchBuilder(self.tokenizer.text, self.tokenizer)
        all_tokens = tokenize_text(self.tokenizer.text)
        chunks = batch_builder.create_chunks(all_tokens)
        training_batches, validation_batches = batch_builder.create_batches(chunks)
        total_batches = len(training_batches)

        print(f"Number of batches to train: {total_batches}")

        # Training loop with multiple epochs
        for epoch in range(1, NB_EPOCHS + 1):
            print(f"Epoch {epoch}/{NB_EPOCHS}")
            np.random.shuffle(training_batches)

            avg_train_loss = 0.0
            nb_batches = 0
            for i, batch in enumerate(training_batches):
                train_loss = self.train_step(batch[0], batch[1])
                avg_train_loss += train_loss
                nb_batches += 1

                # Display progress every 1% of the total batches
                if i % max(1, total_batches // 100) == 0:
                    validation_loss = self.validation_step(batch[0], batch[1])
                    avg_train_loss /= nb_batches
                    percent = (i / total_batches) * 100
                    sys.stdout.write(f"\r[{i}/{total_batches}] {percent:.1f}% - Loss: {avg_train_loss:.4f} - Val Loss: {validation_loss:.4f}")
                    sys.stdout.flush()
                    self.logger.train_log(epoch, i, avg_train_loss, np.exp(avg_train_loss))
                    self.logger.validation_log(epoch, i, validation_loss, np.exp(validation_loss))
                    avg_train_loss = 0.0
                    nb_batches = 0
                    self.save_model()
        
        print("\nModel saved successfully.")


    def save_model(self):
        """ Save the model parameters and vocabulary to files. """
        
        model_parameters = self.model.get_parameters()
        if not MODEL_PATH.parent.exists():
            MODEL_PATH.parent.mkdir(parents=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model_parameters, f)

        vocabulary = self.tokenizer.table_manager.load_table()
        if not VOCABULARY_PATH.parent.exists():
            VOCABULARY_PATH.parent.mkdir(parents=True)
        with open(VOCABULARY_PATH, "w") as f:
            json.dump(vocabulary, f, indent=4)