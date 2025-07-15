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
    vocab = {}

    while True:
        frequencies = {}
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair in frequencies:
                    frequencies[pair] += 1
                else:
                    frequencies[pair] = 1
        
        most_frequent_pair = max(frequencies, key=frequencies.get)
        if frequencies[most_frequent_pair] <= 1:
            break

        for seq in sequences :
            i = 0
            while i < len(seq) - 1:
                pair = (seq[i], seq[i + 1])
                if pair == most_frequent_pair:
                    seq[i] = seq[i] + seq[i+1]
                    seq.pop(i + 1)
                else:
                    i += 1
    
        vocab = {}
        for seq in sequences:
            for token in seq:
                if token not in vocab:
                    vocab[token] = len(vocab)

        if len(vocab) > MAX_TOKENS:
            break
        else:
            print(len(vocab))
    
    table_manager.save_table(vocab)