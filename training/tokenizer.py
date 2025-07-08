from paths import *
import json


def load_table(table_path = TABLE_PATH) -> dict:
    """ Load the association table from a JSON file. """

    if not (table_path).exists():
        return {}
    
    with open(table_path, 'r') as file:
        table = json.load(file)
    return table


def save_table(table: dict, table_path = TABLE_PATH) -> dict:
    """ Save the association table to a JSON file. """

    with open(table_path, 'w') as file:
        json.dump(table, file)


def encode(text : str) -> list:
    """ Convert a text to a numeric vector. """

    chars = list(text)
    vector = []
    association_table = load_table()

    for c in chars:
        if not c in association_table.keys():
            if association_table != {}:
                new_id = int(max(association_table.values())) + 1 
            else:
                new_id = 0

            association_table[c] = new_id
            save_table(association_table)
        
        vector.append(association_table[c])
    
    return vector


def decode(vector : list) -> str:
    """ Convert a numeric vector to a text. """

    table = load_table()
    text = ""

    for nb in vector:
        for key, value in table.items():
            if value == nb:
                text += key
    
    return text