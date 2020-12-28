from pathlib import Path
import keras.layers as layers
import keras.optimizers as optimizers
import keras.metrics as metrics
import keras.regularizers as regularizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys
import yaml

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
from Code.helper import read_data

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']


def create_model(features_num: int, learning_rate: float, regularization_factor: float, hidden_units: list):
    input_layer = layers.Input(shape=(features_num,))
    dense_layer_1 = layers.Dense(hidden_units[0], activation='relu',
                                 kernel_regularizer=regularizers.l2(regularization_factor))(input_layer)
    dense_layer_2 = layers.Dense(hidden_units[1], activation='relu',
                                 kernel_regularizer=regularizers.l2(regularization_factor))(dense_layer_1)
    output_layer = layers.Dense(hidden_units[2], activation='sigmoid',
                                kernel_regularizer=regularizers.l2(regularization_factor))(dense_layer_2)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizers.Adam(lr=learning_rate), loss=params['loss'],
                  metrics=[metrics.RootMeanSquaredError(),
                           metrics.MeanSquaredLogarithmicError(),
                           metrics.MeanAbsoluteError()])
    model.summary()
    return model


def train(model: Model, x_train, y_train):
    model_check_point_cb = ModelCheckpoint(filepath=str(Config.MODEL_FILE),
                                           monitor='val_loss',
                                           save_best_only=True)
    early_stop_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00001, patience=8)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=2, factor=0.1)
    history = model.fit(
        x=x_train,
        y=y_train,
        validation_split=float(params['validation_split']),
        epochs=int(params['epochs']),
        batch_size=int(params['batch_size']),
        verbose=1,
        callbacks=[
            model_check_point_cb,
            early_stop_cb,
            reduce_lr_cb
        ]
    )
    return history


def plot_learning_curves(history):
    plt.plot(history.history['root_mean_squared_error'])
    plt.plot(history.history['val_root_mean_squared_error'])
    plt.title('Model RMSE')
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(str(Config.PLOTS_PATH / 'learning_curve_rmse.png'))
    plt.show()
    plt.clf()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss (MAE)')
    plt.ylabel('Loss (MAE)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(str(Config.PLOTS_PATH / 'learning_curve_loss.png'))
    plt.show()
    plt.clf()


def main():
    x_train, x_test, y_train, y_test = read_data()
    features_num = x_train.shape[1]
    model = create_model(
        features_num,
        learning_rate=float(params['learning_rate']),
        regularization_factor=float(params['regularization_factor']),
        hidden_units=params['hidden_units']
    )
    history = train(model, x_train, y_train)
    plot_learning_curves(history)


main()
