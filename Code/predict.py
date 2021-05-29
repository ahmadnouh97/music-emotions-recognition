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
import Code.modeling as modeling
import Code.preprocessing as preprocessing

learning_rate = modeling.learning_rate
CLASS_NUM = modeling.CLASS_NUM


def read_data():
    x_test_happy = pd.read_csv(str(Config.HAPPY_DATA / 'x_test.csv'), index_col=False)
    x_test_sad = pd.read_csv(str(Config.SAD_DATA / 'x_test.csv'), index_col=False)
    x_test_loud = pd.read_csv(str(Config.LOUD_DATA / 'x_test.csv'), index_col=False)

    y_test_happy = pd.read_csv(str(Config.HAPPY_DATA / 'y_test.csv'), index_col=False)
    y_test_sad = pd.read_csv(str(Config.SAD_DATA / 'y_test.csv'), index_col=False)
    y_test_loud = pd.read_csv(str(Config.LOUD_DATA / 'y_test.csv'), index_col=False)

    test_features = pd.concat([x_test_happy, x_test_sad, x_test_loud])
    test_labels = pd.concat([y_test_happy, y_test_sad, y_test_loud])

    test_features.reset_index(drop=True, inplace=True)
    test_labels.reset_index(drop=True, inplace=True)

    # shuffle
    idx = np.random.permutation(test_features.index)
    test_features = test_features.reindex(idx, axis=0)
    test_labels = test_labels.reindex(idx, axis=0)

    test_features.reset_index(drop=True, inplace=True)
    test_labels.reset_index(drop=True, inplace=True)

    return test_features, test_labels


x_test, y_test = read_data()
encoder = preprocessing.load_encoder(str(Config.MODEL_PATH / 'text_encoder.pkl'))

labels = np.array(list(y_test['label']))
lyrics = np.array(list(x_test['lyrics']))
sound_features = np.array(x_test[[col for col in x_test.columns if col != 'lyrics']])

text_sequences = list(preprocessing.tokenize(lyrics))
print(f'text_sequences = {text_sequences[0]}')
sequences = preprocessing.text_to_sequences(text_sequences, encoder)
print(f'sequences = {sequences[0]}')

sequences = preprocessing.pad_seqs(sequences, max_seq_len=encoder['max_seq_len'])

checkpoint_path = str(Config.MODEL_PATH / Path('checkpoint') / 'cp.ckpt')

model = tf.keras.models.load_model(checkpoint_path,
                                   custom_objects={"F1Score": tfa.metrics.F1Score}, compile=False)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=[tfa.metrics.F1Score(num_classes=CLASS_NUM)])

print(model.summary())

predictions = model.predict([sequences, sound_features])

print(f'predictions_shape = {predictions.shape}')

print(f'prediction = {predictions[0]}')

print(f'label = {labels[0]}')
