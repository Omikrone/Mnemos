from multiprocessing import Pool
from paths import *
import numpy as np

from training.bpe import TokensTableManager, encode_text


CHUNK_SIZE = 64
BATCH_SIZE = 8



class Tokenizer:

    table_manager : TokensTableManager
    text : str

    def __init__(self, text):
        self.text = text
        self.table_manager = TokensTableManager(Path("training/table.json"))


    def decode(self, vector : list) -> str:
        """ Convert a numeric vector to a text. """

        table = self.table_manager.load_table()
        text = ""

        for nb in vector:
            for key, value in table.items():
                if value == nb:
                    text += key
        
        return text


    def create_chunks(self) -> list[list]:
        """ Split the text into inputs and targets of CHUNK_SIZE tokens """

        lines = [line.strip() for line in self.text.split("\n") if line.strip()]
        with Pool() as pool:
            all_tokens = pool.map(encode_text, lines)
        all_tokens = [token for sublist in all_tokens for token in sublist]
        print(f"Number of tokens: {len(all_tokens)}")
        print(f"All tokens: {all_tokens[:10]}")  # Afficher les 10 premiers tokens pour vérification

        nb_chunks = len(all_tokens) // CHUNK_SIZE

        chunks = []
        for i in range(nb_chunks):
            chunks.append(
                (all_tokens[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE], # input
                all_tokens[i*CHUNK_SIZE + 1 : (i+1)*CHUNK_SIZE + 1], # target
                ))

        return chunks


    def create_batches(self, chunks : list) -> list[tuple[np.ndarray, np.ndarray]]:
        
        batches = list()
        nb_batches = len(chunks) // BATCH_SIZE
        for i in range(nb_batches):
            inputs = [couple[0] for couple in chunks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]]
            targets = [couple[1] for couple in chunks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]]

            inputs_batch = np.array(inputs)
            targets_batch = np.array(targets)
            batches.append((inputs_batch, targets_batch))
        
        return batches