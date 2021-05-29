import sys
import yaml
from pathlib import Path
import tensorflow as tf
import tensorflow_addons as tfa

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']

embedd_size = int(params['embedd_size'])
learning_rate = float(params['learning_rate'])
regularization_factor = float(params['regularization_factor'])
dropout_factor = float(params['dropout'])
hidden_units = params['hidden_units']
batch_size = int(params['batch_size'])
epochs = int(params['epochs'])
validation_split = float(params['validation_split'])
CLASS_NUM = 3


def build_model(features_num, vocab_size, max_seq_len):
    lyrics_input = tf.keras.Input(shape=(max_seq_len,), name='lyrics_input')
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedd_size,
        # Use masking to handle the variable sequence lengths
        # mask_zero=True
    )(lyrics_input)

    bi_lstm_layer = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(int(hidden_units[0]),
                             recurrent_dropout=0,
                             activation='tanh',
                             kernel_regularizer=tf.keras.regularizers.L1L2(regularization_factor),
                             activity_regularizer=tf.keras.regularizers.L1L2(regularization_factor)), )(embedding_layer)

    sound_features_input = tf.keras.Input(shape=(features_num,), name='sound_input')

    dense_layer_0 = tf.keras.layers.Dense(int(hidden_units[1]), activation='tanh')(sound_features_input)

    concat_layer = tf.keras.layers.Concatenate(axis=1)([dense_layer_0, bi_lstm_layer])

    dense_layer_1 = tf.keras.layers.Dense(int(hidden_units[2]), activation='tanh',
                                          kernel_regularizer=tf.keras.regularizers.L1L2(regularization_factor),
                                          activity_regularizer=tf.keras.regularizers.L1L2(regularization_factor)
                                          )(concat_layer)

    dropout_layer = tf.keras.layers.Dropout(dropout_factor)(dense_layer_1)

    dense_layer_2 = tf.keras.layers.Dense(int(hidden_units[3]), activation='tanh')(dropout_layer)

    output_layer = tf.keras.layers.Dense(CLASS_NUM, activation='softmax')(dense_layer_2)

    model = tf.keras.models.Model(inputs=[lyrics_input, sound_features_input], outputs=output_layer)

    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate),
                  metrics=[tfa.metrics.F1Score(num_classes=CLASS_NUM)])

    return model

