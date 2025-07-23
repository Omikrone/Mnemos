import sys
import numpy as np
import pickle
from pathlib import Path

from training.tokenizer.parallel_encoding import encode, tokenize_text
from training.tokenizer.bpe import BPETokenizer
from model.transformer_model import TransformerModel
from config.paths import MODEL_PATH, VOCABULARY_PATH
from config.params import MAX_SEQUENCE_LENGTH


class Inference:
    """ Inference class for generating text using a trained Transformer model. """

    model: TransformerModel


    def __init__(self):
        """ Initialize the Inference with the trained model and tokenizer. """

        if not MODEL_PATH.exists() or not VOCABULARY_PATH.exists():
            print("Model or vocabulary file not found. Please train the model first.")
            sys.exit(1)
        self.model = self.load_model(model_path=MODEL_PATH, vocab_path=VOCABULARY_PATH)


    def generate(self, prompt: str, max_length: int = 50) -> str:
        """ Generate text based on the input prompt. """
        
        self.tokenizer = BPETokenizer(prompt)
        tokens = np.array([tokenize_text(prompt)]) # (1, seq_len)
        B, T, D = tokens.shape
        tokens = tokens.reshape(B*T, D)

        for _ in range(max_length):
            if tokens.shape[1] >= MAX_SEQUENCE_LENGTH:
                break
            generated_token_id = self.model.predict_next_token(tokens)  # attend (batch, seq_len)
            generated_token_id = np.array([[generated_token_id]])
            
            tokens = np.concatenate((tokens, generated_token_id), axis=1)
        
        char = self.tokenizer.decode(tokens[0])
        return char


    def load_model(self, model_path: Path, vocab_path: Path) -> TransformerModel:
        """ Load the model from a file. """

        with open(model_path, "rb") as f:
            model_params = pickle.load(f)

        return TransformerModel.from_params(model_params)