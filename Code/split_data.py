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
PREPARED_DATA_PATH = Path(sys.argv[2])


def split_data():
    data = pd.read_csv(FEATURES_FILE, index_col=False)
    y = data['label']
    data_cols = [col for col in data.columns if col != 'music_id' and col != 'label']
    x = data[data_cols]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train.to_csv(str(PREPARED_DATA_PATH / 'x_train.csv') ,index=False)
    x_test.to_csv(str(PREPARED_DATA_PATH / 'x_test.csv') ,index=False)
    y_train.to_csv(str(PREPARED_DATA_PATH / 'y_train.csv') ,index=False)
    y_test.to_csv(str(PREPARED_DATA_PATH / 'y_test.csv') ,index=False)

PREPARED_DATA_PATH.mkdir(parents=True, exist_ok=True)

split_data()