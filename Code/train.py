from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import sys
import yaml

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
from Code.helper import read_data

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']


def main():
    x_train, x_test, y_train, y_test = read_data()
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth']))
    ])
    model.fit(x_train, y_train)

    with open(str(Config.MODEL_PATH / 'model.pkl'), 'wb') as file:
        pkl.dump(model, file)


main()
