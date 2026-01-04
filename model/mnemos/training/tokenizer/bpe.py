from collections import defaultdict

from mnemos.training.tokenizer.vocabulary import VocabularyManager
from mnemos.config.paths import MERGES_PATH, VOCABULARY_PATH
from mnemos.config.params import NB_MAX_TOKENS


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
        reverse_table = {v: k for k, v in table.items()}
        text = ""
        for nb in vector:
            if nb in reverse_table:
                text += reverse_table[nb]
        return text