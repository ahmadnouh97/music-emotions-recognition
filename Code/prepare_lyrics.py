import pandas as pd
import os
import sys
from pathlib import Path
import pyarabic.araby as ar
import re

sys.path.append(str(Path.absolute(self=Path('./'))))
from tqdm import tqdm

ARABIC_LETTERS = [
    'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س',
    'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', ' '
]

NON_ARABIC_LETTER = re.compile('[^' + ''.join(ARABIC_LETTERS + [' ']) + ']')


def read_txt_file(filename: str):
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
    return text


def save_txt_file(filename: str, text: str):
    with open(filename, 'w', encoding="utf8") as file:
        file.write(text)


def preprocessing(text: str):
    paragraphs = text.split('\n')
    processed_paragraphs = []
    for paragraph in paragraphs:
        # trim paragraph
        paragraph = paragraph.strip()
        # strip shadda
        paragraph = ar.strip_shadda(paragraph)
        # strip tashkeel
        paragraph = ar.strip_tashkeel(paragraph)
        # strip tatweel
        paragraph = ar.strip_tatweel(paragraph)
        # strip non arabic letter
        paragraph = re.sub(NON_ARABIC_LETTER, ' ', paragraph)
        # strip multiple spaces with single one
        paragraph = ' '.join(paragraph.split())
        if len(paragraph) > 0: processed_paragraphs.append(paragraph)

    return "\n".join(processed_paragraphs)


def process_lyrics(files: list, verbose=True):
    if verbose:
        print('Start preprocessing lyrics files..\nPlease be patient :)')

    for file in tqdm(files):
        lyrics = read_txt_file(str(LYRICS_PATH / file))
        processed_lyrics = preprocessing(lyrics)
        save_txt_file(str(PROCESSED_LYRICS_PATH / file), processed_lyrics)


LYRICS_PATH = Path(sys.argv[1])
PROCESSED_LYRICS_PATH = Path(sys.argv[2])
PROCESSED_LYRICS_PATH.mkdir(parents=True, exist_ok=True)

lyrics_files = os.listdir(str(LYRICS_PATH))
process_lyrics(lyrics_files)
