import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import pickle as pkl

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config

FEATURES_FILE = sys.argv[1]
ANNOTATIONS_FILE = sys.argv[2]


def create_folders():
    Config.DATA_PATH_PREPARED.mkdir(parents=True, exist_ok=True)


def prepare_data():
    features = pd.read_csv(FEATURES_FILE, index_col=False)
    annotations = pd.read_csv(ANNOTATIONS_FILE, index_col=False)

    data = pd.merge(features, annotations, on='music_id')
    features_cols = [col for col in features.columns if col != 'music_id']

    features = data[features_cols]
    arousal_mean = data['Arousal(mean)']
    valence_mean = data['Valence(mean)']

    # print(len(features))

    x = np.array(features)
    # y1 = np.array(arousal_mean).reshape(-1, 1)
    # y2 = np.array(valence_mean).reshape(-1, 1)
    # y = np.hstack([y1, y2])
    y = np.array(arousal_mean)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    with open(str(Config.DATA_PATH_PREPARED / 'x_train.pkl'), 'wb') as file:
        pkl.dump(x_train, file)
    with open(str(Config.DATA_PATH_PREPARED / 'x_test.pkl'), 'wb') as file:
        pkl.dump(x_test, file)
    with open(str(Config.DATA_PATH_PREPARED / 'y_train.pkl'), 'wb') as file:
        pkl.dump(y_train, file)
    with open(str(Config.DATA_PATH_PREPARED / 'y_test.pkl'), 'wb') as file:
        pkl.dump(y_test, file)


def main():
    create_folders()
    prepare_data()


main()
