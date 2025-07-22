# Parameters for the model and training


## Tokenizer parameters
NB_MAX_TOKENS = 1024            # Maximum number of tokens in the vocabulary

## Batch parameters
MAX_SEQUENCE_LENGTH = 64        # Maximum length of sequences in batches
CHUNK_SIZE = 64                 # Size of chunks to split the text into for training
BATCH_SIZE = 8                  # Number of sequences in a batch

## Training parameters
LEARNING_RATE = 1e-3            # Learning rate for the optimizer
NB_EPOCHS = 1                   # Number of epochs for training

## Model parameters
EMBEDDING_DIM = 64              # Dimension of the embedding vectors
HIDDEN_DIM = 128                # Dimension of the hidden layers

## Layer parameters
EPS = 1e-5                      # Small value to avoid division by zero in layer normalization