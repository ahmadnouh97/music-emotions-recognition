import pickle as pkl
from pathlib import Path
import sys

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config


def read_data():
    with open(str(Config.DATA_PATH_PREPARED / 'x_train.pkl'), 'rb') as file:
        x_train = pkl.load(file)
    with open(str(Config.DATA_PATH_PREPARED / 'x_test.pkl'), 'rb') as file:
        x_test = pkl.load(file)
    with open(str(Config.DATA_PATH_PREPARED / 'y_train.pkl'), 'rb') as file:
        y_train = pkl.load(file)
    with open(str(Config.DATA_PATH_PREPARED / 'y_test.pkl'), 'rb') as file:
        y_test = pkl.load(file)

    return x_train, x_test, y_train, y_test
