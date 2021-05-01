import sys
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_addons as tfa
tf.get_logger().setLevel('WARNING')

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config


params = yaml.safe_load(open(Config.PARAMS_PATH))['train']

vocab_size = int(params['vocab_size'])
embedd_size = int(params['embedd_size'])
learning_rate = float(params['learning_rate'])
regularization_factor = float(params['regularization_factor'])
dropout_factor = float(params['dropout'])
hidden_units = params['hidden_units']
batch_size = int(params['batch_size'])
epochs = int(params['epochs'])
validation_split = float(params['validation_split'])
CLASS_NUM = 3

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def read_data():
    x_train_happy = pd.read_csv(str(Config.HAPPY_DATA / 'x_train.csv'), index_col=False)
    x_train_sad = pd.read_csv(str(Config.SAD_DATA / 'x_train.csv'), index_col=False)
    x_train_loud = pd.read_csv(str(Config.LOUD_DATA / 'x_train.csv'), index_col=False)

    y_train_happy = pd.read_csv(str(Config.HAPPY_DATA / 'y_train.csv'), index_col=False)
    y_train_sad = pd.read_csv(str(Config.SAD_DATA / 'y_train.csv'), index_col=False)
    y_train_loud = pd.read_csv(str(Config.LOUD_DATA / 'y_train.csv'), index_col=False)

    x_train = pd.concat([x_train_happy, x_train_sad, x_train_loud])
    y_train = pd.concat([y_train_happy, y_train_sad, y_train_loud])

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # shuffle
    idx = np.random.permutation(x_train.index)
    x_train = x_train.reindex(idx, axis=0)
    y_train = y_train.reindex(idx, axis=0)

    x_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    return x_train, y_train


def fit_encoder(lyrics):
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=vocab_size
    )

    encoder.adapt(lyrics)
    return encoder


def build_model(encoder, features_num):
    lyrics_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='lyrics_input')
    encoding_layer = encoder(lyrics_input)
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size + 1,
        output_dim=embedd_size,
        # Use masking to handle the variable sequence lengths
        mask_zero=True)(encoding_layer)

    bi_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(int(hidden_units[0]),
                             recurrent_dropout=0.1,
                             kernel_regularizer=tf.keras.regularizers.L1L2(regularization_factor),
                             activity_regularizer=tf.keras.regularizers.L1L2(regularization_factor)),)(embedding_layer)

    sound_features_input = tf.keras.Input(shape=(features_num,), name='sound_input')

    dense_layer_0 = tf.keras.layers.Dense(int(hidden_units[1]), activation='sigmoid')(sound_features_input)

    concat_layer = tf.keras.layers.Concatenate(axis=1)([dense_layer_0, bi_lstm_layer])

    dense_layer_1 = tf.keras.layers.Dense(int(hidden_units[2]), activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.L1L2(regularization_factor),
                                          activity_regularizer=tf.keras.regularizers.L1L2(regularization_factor)
                                          )(concat_layer)

    dropout_layer = tf.keras.layers.Dropout(dropout_factor)(dense_layer_1)

    dense_layer_2 = tf.keras.layers.Dense(int(hidden_units[3]), activation='relu')(dropout_layer)

    output_layer = tf.keras.layers.Dense(CLASS_NUM, activation='softmax')(dense_layer_2)

    model = tf.keras.models.Model(inputs=[lyrics_input, sound_features_input], outputs=output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=[tfa.metrics.F1Score(num_classes=CLASS_NUM)])

    return model



def main():
    x_train, y_train = read_data()
    lyrics = np.array(list(x_train['lyrics']))
    labels = np.array(list(y_train['label']))

    sound_features = np.array(x_train[[col for col in x_train.columns if col != 'lyrics']])
    encoder = fit_encoder(lyrics)

    model = build_model(encoder, len(x_train.columns)-1)
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    labels = tf.keras.utils.to_categorical(labels, CLASS_NUM)

    checkpoint_filepath = str(Config.MODEL_PATH / 'checkpoint')
    model.fit([lyrics, sound_features],
              labels,
              validation_split=validation_split,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      filepath=checkpoint_filepath,
                      save_weights_only=True,
                      monitor='val_loss',
                      mode='min',
                      save_best_only=True)
              ])

    print('save text encoder..')
    pickle.dump({'config': encoder.get_config(),
                 'weights': encoder.get_weights()}
                , open(str(Config.MODEL_PATH / 'text_encoder.pkl'), "wb"))





main()