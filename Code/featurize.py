import pandas as pd
import opensmile
import os
import sys
from pathlib import Path

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
from tqdm import tqdm

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

SONGS_WAV_PATH = Path(sys.argv[1])
FEATURES_FILE = Path(sys.argv[2])


def extract_features(files, extract_to, verbose=True):
    features_df = pd.DataFrame()

    if verbose:
        print('Start extracting features from songs..\nPlease be patient :)')

    with open(str(SONGS_WAV_PATH / 'failed.txt'), 'w', encoding="utf-8") as file_stream:
        for file in tqdm(files):
            try:
                df = smile.process_file(str(SONGS_WAV_PATH / file), channel=1)
            except ValueError:
                df = smile.process_file(str(SONGS_WAV_PATH / file), channel=0)
            finally:
                dot_index = file.rfind('.')
                extension = file[dot_index:]
                filename = file[:len(file) - len(extension)]
                df['music_id'] = filename
                features_df = pd.concat([features_df, df], ignore_index=True)

    features_df.to_csv(extract_to, index=False)


# Config.FEATURES_PATH.mkdir(parents=True, exist_ok=True)

extract_features(
    files=os.listdir(str(SONGS_WAV_PATH)),
    extract_to=str(FEATURES_FILE)
)
