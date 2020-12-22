from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from pathlib import Path
import sys
import json
import yaml
import pickle as pkl

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
from Code.helper import read_data

METRICS_PATH = Path(sys.argv[1])
params = yaml.safe_load(open(Config.PARAMS_PATH))['train']


def evaluate(x_test, y_test):
    with open(str(Config.MODEL_PATH / 'model.pkl'), 'rb') as file:
        model = pkl.load(file)

    predictions = model.predict(x_test)
    rmsle = mean_squared_log_error(y_true=y_test, y_pred=predictions)
    rmse = mean_squared_error(y_true=y_test, y_pred=predictions)
    mae = mean_absolute_error(y_true=y_test, y_pred=predictions)

    with open(METRICS_PATH, 'w') as file:
        json.dump({
            'RMSE': rmse,
            'RMSLE': rmsle,
            'MAE': mae
        }, file)

    print(f'predictions = {predictions[:10]}')
    print(f'y_true = {y_test[:10]}')


def main():
    _, x_test, _, y_test = read_data()
    evaluate(x_test, y_test)


main()
