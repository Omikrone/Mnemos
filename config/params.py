# Parameters for the model and training


## Tokenizer parameters
NB_MAX_TOKENS = 1024            # Maximum number of tokens in the vocabulary

## Batch parameters
MAX_SEQUENCE_LENGTH = 64        # Maximum length of sequences in batches
CHUNK_SIZE = 64                 # Size of chunks to split the text into for training
BATCH_SIZE = 8                  # Number of sequences in a batch

## Training parameters
LEARNING_RATE = 5e-4            # Learning rate for the optimizer
NB_EPOCHS = 3                   # Number of epochs for training

## Model parameters
EMBEDDING_DIM = 128             # Dimension of the embedding vectors
HIDDEN_DIM = 256                # Dimension of the hidden layers
NB_ATTENTION_HEADS = 2          # Number of self-attention heads
NB_LAYERS = 2                   # Number of layers in the Transformer model

## Layer parameters
EPS = 1e-5                      # Small value to avoid division by zero in layer normalization