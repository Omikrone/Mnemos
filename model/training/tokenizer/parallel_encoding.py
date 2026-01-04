import heapq
from multiprocessing import Pool, cpu_count
import time

from training.tokenizer.vocabulary import VocabularyManager
from config.paths import MERGES_PATH, VOCABULARY_PATH


global_table = None
global_merges = None
global_merge_rank = None


def init_worker():
    global global_table, global_merges, global_merge_rank
    table_manager = VocabularyManager(VOCABULARY_PATH, MERGES_PATH)
    global_table = table_manager.load_table()
    merges = table_manager.load_merges()
    global_merges = merges
    global_merge_rank = {tuple(pair): i for i, pair in enumerate(merges)}


def encode(text: str) -> list[int]:
    """ Encode a text into a list of integers using BPE. """

    global global_table, global_merge_rank

    merge_rank = global_merge_rank

    # Initial sequence: characters with a space prefix for consistency
    seq = [' ' + text[0]] + list(text[1:]) if text else []

    # Build initial list of pairs with priority
    pairs = []
    for i in range(len(seq) - 1):
        pair = (seq[i], seq[i + 1])
        if pair in merge_rank:
            heapq.heappush(pairs, (merge_rank[pair], i, pair))
    
    while pairs:
        _, i, pair = heapq.heappop(pairs)
        
        # Validate the pair is still at the right position
        if i >= len(seq) - 1 or (seq[i], seq[i + 1]) != pair:
            continue
        
        # Merge the pair
        merged = seq[i] + seq[i + 1]
        seq[i] = merged
        del seq[i + 1]
        
        # Reinsert surrounding pairs
        if i > 0:
            prev = (seq[i - 1], seq[i])
            if prev in merge_rank:
                heapq.heappush(pairs, (merge_rank[prev], i - 1, prev))
        if i < len(seq) - 1:
            nxt = (seq[i], seq[i + 1])
            if nxt in merge_rank:
                heapq.heappush(pairs, (merge_rank[nxt], i, nxt))
    
    tokens = []
    for c in seq:
        if c in global_table:
            tokens.append(global_table[c])
        else:            
            print(f"Warning: Character '{c}' not found in vocabulary. Skipping.")
            tokens.append(0)
            continue
    return tokens


def tokenize_text(text: str) -> list[list[int]]:
    """ Tokenize the text using the BPE tokenizer. """

    lines = text.splitlines()
    start_time = time.time()
    with Pool(cpu_count(), initializer=init_worker) as pool:
        all_tokens = pool.map(encode, lines)
        pool.close()
        pool.join()
    elapsed_time = time.time() - start_time

    print(f"Tokenizing completed in {elapsed_time:.4f} seconds.")
    return all_tokens