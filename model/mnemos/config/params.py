# Parameters for the model and training


## Tokenizer parameters
NB_MAX_TOKENS = 1024            # Maximum number of tokens in the vocabulary

## Batch parameters
MAX_SEQUENCE_LENGTH = 64       # Maximum length of sequences in batches
CHUNK_SIZE = 64                # Size of chunks to split the text into for training
BATCH_SIZE = 4                  # Number of sequences in a batch

## Training parameters
LEARNING_RATE = 1e-3            # Learning rate for the optimizer
NB_EPOCHS = 7                   # Number of epochs for training

## Model parameters
EMBEDDING_DIM = 256             # Dimension of the embedding vectors
HIDDEN_DIM = 1024               # Dimension of the hidden layers
NB_ATTENTION_HEADS = 4          # Number of self-attention heads
NB_LAYERS = 1                   # Number of layers in the Transformer model
DROPOUT_RATE = 0.1              # Dropout rate for regularization

## Layer parameters
EPS = 1e-6                      # Small value to avoid division by zero in layer normalization