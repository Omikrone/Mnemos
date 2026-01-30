import sys
import pickle
from mnemos import xp
import numpy as np

from pathlib import Path

from mnemos.training.tokenizer.parallel_encoding import tokenize_text
from mnemos.training.tokenizer.bpe import BPETokenizer
from mnemos.transformer.transformer_model import TransformerModel
from mnemos.config.paths import MODEL_PATH, VOCABULARY_PATH
from mnemos.config.params import MAX_SEQUENCE_LENGTH


class Inference:
    """ Inference class for generating text using a trained Transformer model. """

    model: TransformerModel


    def __init__(self):
        """ Initialize the Inference with the trained model and tokenizer. """

        if not MODEL_PATH.exists() or not VOCABULARY_PATH.exists():
            print("Model or vocabulary file not found. Please train the model first.")
            sys.exit(1)
        self.model = self.load_model(model_path=MODEL_PATH, vocab_path=VOCABULARY_PATH)


    def generate(self, prompt: str, max_length: int = 300) -> str:
        """ Generate text based on the input prompt. """
        
        self.tokenizer = BPETokenizer(prompt)
        tokens = np.array([tokenize_text(prompt)]) # (1, seq_len)
        B, T, D = tokens.shape
        tokens = tokens.reshape(B*T, D)

        for _ in range(max_length):
            if tokens.shape[1] >= max_length:
                break

            context = tokens[:, -MAX_SEQUENCE_LENGTH:]  if tokens.shape[1] > MAX_SEQUENCE_LENGTH else tokens
            generated_token_id = self.predict_next_token(context)  # attend (batch, seq_len)
            generated_token_id = np.array([[generated_token_id]])
            
            tokens = np.concatenate((tokens, generated_token_id), axis=1)
        
        char = self.tokenizer.decode(tokens[0])
        return char


    def predict_next_token(self, input_ids: np.ndarray) -> int:
        """ Predict the next token given the input IDs. """

        logits = self.model.forward(input_ids, train=False)
        if xp != np:
            logits = xp.asnumpy(logits)
        last_logits = logits[0, -1]
        prediction = self.sample_top_k(last_logits, k=10)
        return prediction
    

    def sample_top_k(self, logits, k=10) -> int:
        """ Sample from the top k logits using softmax. """

        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_logits = logits[top_k_indices]
        probs = np.exp(top_k_logits) / np.sum(np.exp(top_k_logits))
        return np.random.choice(top_k_indices, p=probs)


    def load_model(self, model_path: Path, vocab_path: Path) -> TransformerModel:
        """ Load the model from a file. """

        print("Curren directory:", Path.cwd())
        print("Loading model from:", model_path)
        with open(model_path, "rb") as f:
            model_params = pickle.load(f)

        return TransformerModel.from_params(model_params)