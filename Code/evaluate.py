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

learning_rate = modeling.learning_rate
CLASS_NUM = modeling.CLASS_NUM


def read_data():
    happy_features = pd.read_csv(str(Config.HAPPY_DATA / 'features.csv'), index_col=False)
    sad_features = pd.read_csv(str(Config.SAD_DATA / 'features.csv'), index_col=False)

    x_test_sad = pd.read_csv(str(Config.SAD_DATA / 'features.csv'), index_col=False)

    y_test_happy = pd.read_csv(str(Config.HAPPY_DATA / 'features.csv'), index_col=False)
    y_test_sad = pd.read_csv(str(Config.SAD_DATA / 'features.csv'), index_col=False)

    test_features = pd.concat([x_test_happy, x_test_sad])
    test_labels = pd.concat([y_test_happy, y_test_sad])

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

checkpoint_path = str(Config.MODEL_PATH / Path('checkpoint') / 'cp.ckpt')

model = tf.keras.models.load_model(checkpoint_path,
                                   custom_objects={'f1-score': tfa.metrics.F1Score(num_classes=CLASS_NUM)}, compile=False)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate),
              metrics=[tfa.metrics.F1Score(num_classes=CLASS_NUM)])

print(model.summary())
