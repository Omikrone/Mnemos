import unicodedata
import re
import os

from paths import *


MIN_LENGTH_LINE = 10


class PreProcesser:
    """ PreProcesser class for cleaning and normalizing text data. """

    speaker_pattern: re.Pattern

    def __init__(self):
        self.speaker_pattern = re.compile(r"\[.*?\]")


    def __call__(self, data_file_path: Path) -> str:
        """ Clean and normalize the text data from the given file path. """

        cleaned_lines = []

        # Obtenir la taille totale du fichier (en octets)
        total_size = os.path.getsize(data_file_path)
        processed_size = 0
        last_printed_percent = -1  # Pour éviter les doublons

        with open(data_file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():

                # Nettoyage et normalisation de la ligne
                cleaned_line = self._clean_text(line)
                cleaned_line = self._normalize_whitespace(cleaned_line)

                if len(cleaned_line) < MIN_LENGTH_LINE:
                    continue
                cleaned_lines.append(cleaned_line)

                # Mise à jour de la taille traitée
                processed_size += len(line.encode("utf-8"))

                # Affichage du pourcentage (uniquement si changement)
                percent = int((processed_size / total_size) * 100)
                if percent != last_printed_percent:
                    print(f"\rPreprocessing: {percent}%", end="")
                    last_printed_percent = percent

        print("\nPreprocessing finished.")
        return "\n".join(cleaned_lines)
    

    def _clean_text(self, text: str) -> str:
        """ Clean and normalize the text. """

        text = self.speaker_pattern.sub("", text)
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("’", "'").replace("‘", "'")
        text = text.replace("“", '"').replace("”", '"')
        return text.lower()


    def _normalize_whitespace(self, text: str) -> str:
        """ Normalize whitespace in the text. """
        
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text.strip()