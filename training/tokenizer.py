from paths import *
import json
import numpy as np


CHUNK_SIZE = 64
BATCH_SIZE = 8


class TokensTableManager:

    table_path : Path

    def __init__(self, table_path : Path):
        self.table_path = table_path


    def load_table(self) -> dict:
        """ Load the association table from a JSON file. """

        if not (self.table_path).exists():
            return {}
        
        with open(self.table_path, 'r') as file:
            table = json.load(file)
        return table


    def save_table(self, table: dict) -> dict:
        """ Save the association table to a JSON file. """

        with open(self.table_path, 'w') as file:
            json.dump(table, file)



class Tokenizer:

    table_manager : TokensTableManager
    text : str

    def __init__(self, text):
        self.text = text
        self.table_manager = TokensTableManager(Path("training/table.json"))

    def _encode(self, text : str) -> list:
        """ Convert a text to a numeric vector. """

        chars = list(text)
        vector = []
        association_table = self.table_manager.load_table()

        for c in chars:
            if not c in association_table.keys():
                if association_table != {}:
                    new_id = int(max(association_table.values())) + 1 
                else:
                    new_id = 0

                association_table[c] = new_id
                self.table_manager.save_table(association_table)
            
            vector.append(association_table[c])
        
        return vector


    def _decode(self, vector : list) -> str:
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

        tokens = self._encode(self.text)
        nb_chunks = len(tokens) // CHUNK_SIZE

        chunks = []
        for i in range(nb_chunks):
            chunks.append(
                (tokens[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE], # input
                tokens[i*CHUNK_SIZE + 1 : (i+1)*CHUNK_SIZE + 1], # target
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