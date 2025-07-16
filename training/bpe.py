from collections import defaultdict
import json
from pathlib import Path
import pickle
from paths import TABLE_PATH

MAX_TOKENS = 1024


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
            json.dump(table, file, indent=4)

table_manager = TokensTableManager(TABLE_PATH)


def get_vocabulary_size() -> int:
    """ Get the size of the vocabulary. """
    association_table = table_manager.load_table()
    return len(association_table)


def bpe_builder(text: str) -> None:
    """ Build the vocabulary from the text. """

    sequences = [list(' ' + w) for w in text.split(" ")]
    vocab = {char: idx for idx, char in enumerate(sorted(set("".join(text))))}
    occurences = defaultdict(set)
    merge_rules = dict()

    for i, seq in enumerate(sequences):
        for j in range(len(seq) - 1):
            pair = (seq[j], seq[j + 1])
            occurences[pair].add(i)

    while True:
        
        most_frequent_pair = max(occurences, key=lambda p: len(occurences[p]))
        merge_rules[most_frequent_pair] = most_frequent_pair[0] + most_frequent_pair[1]
        if len(occurences[most_frequent_pair]) <= 1:
            break
        
        for seq_i in occurences[most_frequent_pair]:
            #print(seq_i)
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

        if len(vocab) > MAX_TOKENS:
            break
        else:
            print(len(vocab))

        new_token = most_frequent_pair[0] + most_frequent_pair[1]
        if new_token not in vocab:
            vocab[new_token] = len(vocab)
    table_manager.save_table(vocab)
    with open("merges.pkl", "wb") as f:
        pickle.dump(merge_rules, f)


def encode_text(text: str) -> list:
    """ Encode the text into a list of integers. """

    with open("merges.pkl", "rb") as f:
        merge_rules = pickle.load(f)
    table = table_manager.load_table()
    tokens = list()
    
    chars = list(text)

    i = 0
    while i < len(chars) - 1:
        print(f"i: {i} / {len(chars)}")
        merge = (chars[i], chars[i + 1])
        if merge in merge_rules:
            chars[i] = merge_rules[merge]
            del chars[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    for c in chars:
        tokens.append(table[c])
    return tokens