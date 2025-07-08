from training import data_retriever

inputs, targets = data_retriever.create_batches()
print(inputs, "\n\n", targets)