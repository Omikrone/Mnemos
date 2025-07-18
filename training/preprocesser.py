import unicodedata
from paths import *

import re
import os


class PreProcesser:
    def __init__(self):
        self.speaker_pattern = re.compile(r"\[.*?\]")

    def __call__(self, data_file_path: Path) -> str:
        """ Nettoie le fichier de données en supprimant les balises de locuteur. """

        cleaned_lines = []

        # Obtenir la taille totale du fichier (en octets)
        total_size = os.path.getsize(data_file_path)
        processed_size = 0
        last_printed_percent = -1  # Pour éviter les doublons

        with open(data_file_path, "r", encoding="utf-8") as file:
            for line in file:
                processed_size += len(line.encode("utf-8"))  # Mesurer en octets
                cleaned_line = self.speaker_pattern.sub("", line).lower()
                cleaned_line = unicodedata.normalize("NFKC", cleaned_line)
                cleaned_lines.append(cleaned_line)

                # Affichage du pourcentage (uniquement si changement)
                percent = int((processed_size / total_size) * 100)
                if percent != last_printed_percent:
                    print(f"\rPrétraitement : {percent}%", end="")
                    last_printed_percent = percent

        print("\nPrétraitement terminé.")
        return "".join(cleaned_lines)