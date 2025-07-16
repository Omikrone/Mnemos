from model.embeddings import MAX_SEQUENCE_LENGTH
from model.transformer_model import TransformerModel
from training.bpe import encode_text
from training.tokenizer import Tokenizer
import numpy as np
import pickle
from pathlib import Path
import json


class Inference:
    def __init__(self):
        self.model = self.load_model(model_path=Path("save/model.pkl"), vocab_path=Path("save/vocabulary.json"))

    def generate(self, prompt: str, max_length: int = 50) -> str:
        self.tokenizer = Tokenizer(prompt)
        tokens = np.array([encode_text(prompt)])  # (1, seq_len)
        
        for _ in range(max_length):
            if tokens.shape[1] >= MAX_SEQUENCE_LENGTH:
                break
            generated_token_id = self.model.predict_next_token(tokens)  # attend (batch, seq_len)
            generated_token_id = np.array([[generated_token_id]])
            
            tokens = np.concatenate((tokens, generated_token_id), axis=1)
        
        return self.tokenizer.decode(tokens[0])


    def load_model(self, model_path: Path, vocab_path: Path) -> TransformerModel:
        """ Load the model from a file. """

        with open(model_path, "rb") as f:
            model_params = pickle.load(f)
        with open(vocab_path, "r") as f:
            vocab = json.load(f)

        return TransformerModel.from_params(model_params)