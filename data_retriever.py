from paths import *
import tokenizer
import numpy as np

import re

CHUNK_SIZE = 64
BATCH_SIZE = 8
STRIDE_LENGTH = 5


def load_data(data_dir = DATA_DIR):
    """ Load training data """

    with open(data_dir / "assemblee_nationale.txt", "r", encoding='utf-8') as file:
        return file.readlines()
    

def clean_data(lines : list) -> str:
    """ Clean the lines and remove unecessaries caracters """

    full_text = ""
    for line in lines:
        line_without_speaker = re.sub(r"\[.*?\]", "", line)
        full_text += line_without_speaker
    
    return full_text


def create_chunks(tokens : list) -> list[list]:
    """ Split the text into segments of 256 tokens and create the batches """

    nb_chunks = len(tokens) // CHUNK_SIZE

    chunks = []
    for i in range(nb_chunks):
        chunks.append(
            (tokens[i*CHUNK_SIZE : (i+1)*CHUNK_SIZE - 1], # input
             tokens[i*CHUNK_SIZE + 1 : (i+1)*CHUNK_SIZE], # target
             ))

    return chunks

def prepare_data():
    lines = load_data()
    clean_text = clean_data(lines)
    tokens = tokenizer.encode(clean_text)
    
    chunks = create_chunks(tokens)
    
    nb_batches = len(chunks) // BATCH_SIZE
    for i in range(nb_batches):
        inputs = [couple[0] for couple in chunks[i*nb_batches : (i+1)*nb_batches]]
        targets = [couple[1] for couple in chunks[i*nb_batches : (i+1)*nb_batches]]

        inputs_batch = np.array(inputs)
        targets_batch = np.array(targets)
        print(inputs[1])
        input()
        #print(inputs_batch.shape)


prepare_data()