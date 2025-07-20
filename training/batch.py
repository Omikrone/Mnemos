from codecs import encode
from multiprocessing import Pool
from paths import *
import numpy as np

from training.tokenizer import BPETokenizer


CHUNK_SIZE = 64
BATCH_SIZE = 8


class BatchBuilder:
    """ BatchBuilder class for creating batches of training data. """

    tokenizer: BPETokenizer
    text : str


    def __init__(self, text: str, tokenizer: BPETokenizer):
        """ Initialize the BatchBuilder with the text to build the vocabulary. """

        self.text = text
        self.tokenizer = tokenizer


    def create_chunks(self) -> list[list]:
        """ Split the text into inputs and targets of CHUNK_SIZE tokens """

        lines = [line.strip() for line in self.text.split("\n") if line.strip()]
        with Pool() as pool:
            all_tokens = pool.map(encode, lines)
        all_tokens = [token for sublist in all_tokens for token in sublist]
        nb_chunks = len(all_tokens) // CHUNK_SIZE

        chunks = []
        for i in range(nb_chunks):
            chunks.append(
                (all_tokens[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE], # input
                all_tokens[i*CHUNK_SIZE + 1 : (i+1)*CHUNK_SIZE + 1], # target
                ))

        return chunks


    def create_batches(self, chunks : list) -> list[tuple[np.ndarray, np.ndarray]]:
        """ Create batches of BATCH_SIZE from the chunks. """
        
        batches = list()
        nb_batches = len(chunks) // BATCH_SIZE
        for i in range(nb_batches):
            inputs = [couple[0] for couple in chunks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]]
            targets = [couple[1] for couple in chunks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]]

            inputs_batch = np.array(inputs)
            targets_batch = np.array(targets)
            batches.append((inputs_batch, targets_batch))
        
        return batches