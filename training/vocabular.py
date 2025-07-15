from collections import defaultdict
import time
from training.tokenizer import TokensTableManager, TABLE_PATH

table_manager = TokensTableManager(TABLE_PATH)
MAX_TOKENS = 1024


def get_vocabulary_size() -> int:
    """ Get the size of the vocabulary. """
    association_table = table_manager.load_table()
    return len(association_table)


def bpe_builder(text: str) -> None:
    """ Build the vocabulary from the text. """

    sequences = [list(' ' + w) for w in text.split(" ")]
    vocab = defaultdict(int)
    occurences = defaultdict(set)

    for i, seq in enumerate(sequences):
        for j in range(len(seq) - 1):
            pair = (seq[j], seq[j + 1])
            occurences[pair].add(i)

    while True:
        
        most_frequent_pair = max(occurences, key=lambda p: len(occurences[p]))
        print(most_frequent_pair)
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


def encode_text(text: str) -> list:
    """ Encode the text into a list of integers. """
    
    sequences = [list(' ' + w) for w in text.split(" ")]

    while True:
        for seq in sequences:
            for i in range(len(seq)):
                if seq[i] not in table_manager.load_table():
                    raise ValueError(f"Token '{seq[i]}' not found in the vocabulary.")
                