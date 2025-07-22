import json
import pickle
from pathlib import Path


class VocabularyManager:

    table_path : Path
    merges_path : Path

    def __init__(self, table_path : Path, merges_path : Path):
        self.table_path = table_path
        self.merges_path = merges_path


    def load_table(self) -> dict:
        """ Load the association table from a JSON file. """

        if not (self.table_path).exists():
            return {}
        
        with open(self.table_path, 'r') as file:
            table = json.load(file)
        return table
    
    def load_merges(self) -> list:
        """ Load the merge rules from a pickle file. """

        if not self.merges_path.exists():
            return []
        
        with open(self.merges_path, "rb") as f:
            merges = pickle.load(f)
        return merges


    def save_table(self, table: dict) -> None:
        """ Save the association table to a JSON file. """

        if not self.table_path.parent.exists():
            self.table_path.parent.mkdir(parents=True)
        with open(self.table_path, 'w') as file:
            json.dump(table, file, indent=4)


    def save_merges(self, merges: list) -> None:
        """ Save the merge rules to a pickle file. """

        if not self.merges_path.parent.exists():
            self.merges_path.parent.mkdir(parents=True)
        with open(self.merges_path, "wb") as f:
            pickle.dump(merges, f)
