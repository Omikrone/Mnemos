import numpy as np


EMBEDDING_DIMENSION = 64
MAX_SEQUENCE_LENGTH = 64


def create_token_embedding(vocab_size : int, embedding_dim = EMBEDDING_DIMENSION) -> np.ndarray:
    """ Create the matrix for the tokens embedding """

    matrix = list()
    # Random initialization of the tokens embedding
    for _ in range(vocab_size):
        random_embedding = [np.random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
        matrix.append(random_embedding)

    return np.array(matrix)


def create_position_embedding(max_seq_len = MAX_SEQUENCE_LENGTH, embedding_dim = EMBEDDING_DIMENSION) -> np.ndarray:
    """ Create the matrix for the positional embedding """

    matrix = list()
    # Random initialization of the positional embedding
    for _ in range(max_seq_len):
        random_embedding = [np.random.uniform(-0.1, 0.1) for _ in range(embedding_dim)]
        matrix.append(random_embedding)

    return np.array(matrix)


def create_total_embedding(input_batches, target_batches) -> np.ndarray:
    """ Create the input embedding from the token and positional embedding """

    print("TARGETS SHAPE:", target_batches.shape)

    token_embeddings = create_token_embedding(85)
    position_embeddings = create_position_embedding()

    embedded_inputs = token_embeddings[input_batches]
    embedded_inputs += position_embeddings

    print("EMBEDDED INPUTS SHAPE:", embedded_inputs.shape)
    return embedded_inputs