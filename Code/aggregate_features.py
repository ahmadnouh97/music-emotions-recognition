import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle as pkl

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config

FEATURES_PATH = Path(sys.argv[1])
PROCESSED_LYRICS_PATH = Path(sys.argv[2])
LABEL = sys.argv[3]

def get_lyrics(filename: str):
    with open(filename, 'r', encoding="utf8") as file:
        text = file.read()
    return text

def aggregate_data(sound_features_file: str, save_to: str):
    features_df = pd.read_csv(sound_features_file, index_col=False)
    features_df['lyrics'] = features_df['music_id'].apply(lambda music_id: get_lyrics(str(PROCESSED_LYRICS_PATH / music_id) + '.txt'))
    features_df['label'] = LABEL
    features_df.to_csv(save_to, index=False)


FEATURES_PATH.mkdir(parents=True, exist_ok=True)

aggregate_data(
    sound_features_file=str(FEATURES_PATH / 'sound_features.csv'),
    save_to=str(FEATURES_PATH / 'features.csv')
)

