from tensorflow.keras.models import Model, load_model
import tensorflow.keras.metrics as metrics
import tensorflow.keras.optimizers as optimizers
from pathlib import Path
import sys
import json
import yaml

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
from Code.helper import read_data
from Code.train import create_model

METRICS_PATH = Path(sys.argv[1])
params = yaml.safe_load(open(Config.PARAMS_PATH))['train']


def evaluate(model: Model, x_test, y_test):
    results = model.evaluate(x_test, y_test)
    with open(METRICS_PATH, 'w') as file:
        json.dump({
            'Loss': results[0],
            'RMSE': results[1],
            'RMSLE': results[2],
            'MAE': results[3]
        }, file)

    print(f'results = {results}')
    predictions = model.predict(x_test)

    print(f'predictions = {predictions[:10]}')
    print(f'y_true = {y_test[:10]}')


def main():
    _, x_test, _, y_test = read_data()
    features_num = x_test.shape[1]
    # model = create_model(
    #     features_num,
    #     learning_rate=float(params['learning_rate']),
    #     regularization_factor=float(params['regularization_factor']),
    #     hidden_units=params['hidden_units']
    # )
    # model.load_weights(str(Config.MODEL_FILE))
    model = load_model(
        str(Config.MODEL_FILE),
        compile=False
    )
    model.compile(optimizer=optimizers.Adam(lr=float(params['learning_rate'])), loss=params['loss'],
                  metrics=[metrics.RootMeanSquaredError(),
                           metrics.MeanSquaredLogarithmicError(),
                           metrics.MeanAbsoluteError()])
    evaluate(model, x_test, y_test)


main()
