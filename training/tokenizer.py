from collections import defaultdict
import json
from pathlib import Path
import pickle
import heapq

from paths import MERGES_PATH, VOCABULARY_PATH
from params import NB_MAX_TOKENS


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


class BPETokenizer:
    """ Byte Pair Encoding (BPE) class for building and encoding text. """

    text: str
    table_manager: VocabularyManager


    def __init__(self, text: str):
        """ Initialize the BPE class with the text to build the vocabulary. """
        
        self.text = text
        self.table_manager = VocabularyManager(VOCABULARY_PATH, MERGES_PATH)


    def get_vocabulary_size(self) -> int:
        """ Get the size of the vocabulary. """

        association_table = self.table_manager.load_table()
        return len(association_table)


    def build(self) -> None:
        """ Build the vocabulary from given text. """

        print("Building vocabulary, this may take a while...")

        vocab = dict()
        sequences = [list(' ' + w) for w in self.text.split(" ")]

        # Initialize vocabulary with base letters and space
        base_letters = list(set(''.join(self.text)))
        vocab['<unk>'] = 0
        base_letters.extend([' ' + c for c in base_letters])
        for c in base_letters:
            if c not in vocab:
                vocab[c] = len(vocab)
        
        occurences = defaultdict(set)
        merge_rules = list()

        for i, seq in enumerate(sequences):
            for j in range(len(seq) - 1):
                pair = (seq[j], seq[j + 1])
                occurences[pair].add(i)

        while True:
            
            most_frequent_pair = max(occurences, key=lambda p: len(occurences[p]))
            merge_rules.append(most_frequent_pair)
            if len(occurences[most_frequent_pair]) <= 1:
                break
            
            for seq_i in occurences[most_frequent_pair]:
                seq = sequences[seq_i]
                i = 0
                while i < len(seq) - 1:
                    pair = (seq[i], seq[i + 1])
                    if pair == most_frequent_pair:
                        new_token = seq[i] + seq[i + 1]
                        seq[i] = new_token
                        del seq[i + 1]
                        if i > 0:
                            left_pair = (seq[i - 1], new_token)
                            occurences[left_pair].add(seq_i)

                        if i < len(seq) - 1:
                            right_pair = (new_token, seq[i + 1])
                            occurences[right_pair].add(seq_i)
                    else:
                        i += 1
            del occurences[most_frequent_pair]

            if len(vocab) > NB_MAX_TOKENS:
                break

            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            if new_token not in vocab:
                vocab[new_token] = len(vocab)
                
        self.table_manager.save_table(vocab)
        self.table_manager.save_merges(merge_rules)

        print(f"Vocabulary built with {len(vocab)} tokens.")
    

    def decode(self, vector : list) -> str:
        """ Convert a numeric vector to a text. """

        table = self.table_manager.load_table()
        text = ""

        for nb in vector:
            for key, value in table.items():
                if value == nb:
                    text += key
        
        return text


def encode(text: str) -> list[int]:
    """ Encode a text into a list of integers using BPE. """

    table_manager = VocabularyManager(VOCABULARY_PATH, MERGES_PATH)
    table = table_manager.load_table()
    merges = table_manager.load_merges()
    
    merge_rank = {tuple(pair): i for i, pair in enumerate(merges)}
    
    # Initial sequence: characters with a space prefix for consistency
    seq = [' ' + text[0]] + list(text[1:]) if text else []

    # Build initial list of pairs with priority
    pairs = []
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i + 1])
        if pair in merge_rank:
            heapq.heappush(pairs, (merge_rank[pair], i, pair))
    
    while pairs:
        _, i, pair = heapq.heappop(pairs)
        
        # Validate the pair is still at the right position
        if i >= len(seq) - 1 or (seq[i], seq[i + 1]) != pair:
            continue
        
        # Merge the pair
        merged = seq[i] + seq[i + 1]
        seq[i] = merged
        del seq[i + 1]
        
        # Reinsert surrounding pairs
        if i > 0:
            prev = (seq[i - 1], seq[i])
            if prev in merge_rank:
                heapq.heappush(pairs, (merge_rank[prev], i - 1, prev))
        if i < len(seq) - 1:
            nxt = (seq[i], seq[i + 1])
            if nxt in merge_rank:
                heapq.heappush(pairs, (merge_rank[nxt], i, nxt))
    
    tokens = []
    for c in seq:
        if c in table:
            tokens.append(table[c])
        else:            
            print(f"Warning: Character '{c}' not found in vocabulary. Skipping.")
            tokens.append(0)
            continue
    return tokens