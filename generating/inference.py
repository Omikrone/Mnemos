import sys
import numpy as np
import pickle
from pathlib import Path

from training.tokenizer import BPETokenizer
from model.embeddings import MAX_SEQUENCE_LENGTH
from model.transformer_model import TransformerModel


class Inference:
    """ Inference class for generating text using a trained Transformer model. """

    model: TransformerModel


    def __init__(self):
        """ Initialize the Inference with the trained model and tokenizer. """

        if not Path("save/model.pkl").exists() or not Path("save/vocabulary.json").exists():
            print("Model or vocabulary file not found. Please train the model first.")
            sys.exit(1)
        self.model = self.load_model(model_path=Path("save/model.pkl"), vocab_path=Path("save/vocabulary.json"))


    def generate(self, prompt: str, max_length: int = 50) -> str:
        """ Generate text based on the input prompt. """
        
        self.tokenizer = BPETokenizer(prompt)
        tokens = np.array([self.tokenizer.encode(prompt)])  # (1, seq_len)

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