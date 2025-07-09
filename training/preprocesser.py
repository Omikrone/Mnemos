from paths import *

import re


class PreProcesser:

    def __call__(self, data_file_path : Path) -> str:

        with open(data_file_path, "r", encoding='utf-8') as file:
            lines = file.readlines()

        full_text = ""
        for line in lines:
            line_without_speaker = re.sub(r"\[.*?\]", "", line)
            full_text += line_without_speaker
        
        return full_text