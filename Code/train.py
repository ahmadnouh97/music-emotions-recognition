import sys
import yaml
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

tf.get_logger().setLevel('WARNING')
#
sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.config import Config
import Code.modeling as modeling
import Code.preprocessing as preprocessing

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']
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


def main():
    x_train, y_train = read_data()
    lyrics = np.array(list(x_train['lyrics']))
    labels = np.array(list(y_train['label']))

    sound_features = np.array(x_train[[col for col in x_train.columns if col != 'lyrics']])

    encoder, text_sequences = preprocessing.fit_encoder(lyrics, vocab_size=int(params['vocab_size']))
    print(f'word_index = {len(encoder.get("word_index"))}')
    sequences = preprocessing.text_to_sequences(text_sequences, encoder)
    print(f'sequences = {encoder["max_seq_len"]}')

    sequences = preprocessing.pad_seqs(sequences, max_seq_len=encoder['max_seq_len'])
    preprocessing.save_encoder(str(Config.MODEL_PATH / 'text_encoder.pkl'), encoder)
    vocab_size = encoder['vocab_size']
    max_seq_len = encoder['max_seq_len']
    # word_index = encoder['word_index']
    # index_word = encoder['index_word']
    # tokenizer_conditions = encoder['tokenizer_conditions']
    # tokenizer_morphs = encoder['tokenizer_morphs']
    model = modeling.build_model(len(x_train.columns) - 1, vocab_size, max_seq_len)
    tf.keras.utils.plot_model(model, to_file=str(Config.PLOTS_PATH / 'model_architecture.png'), dpi=128, show_shapes=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    labels = tf.keras.utils.to_categorical(labels, modeling.CLASS_NUM)

    checkpoint_path = str(Config.MODEL_PATH / Path('checkpoint') / 'cp.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    model.fit([sequences, sound_features],
              labels,
              validation_split=modeling.validation_split,
              epochs=modeling.epochs,
              batch_size=modeling.batch_size,
              callbacks=[
                  tf.keras.callbacks.ModelCheckpoint(
                      filepath=checkpoint_path,
                      monitor='val_loss',
                      mode='min',
                      save_best_only=True
                  )
              ])

    # print('save text encoder..')
    # pickle.dump({'config': encoder.get_config(),
    #              'weights': encoder.get_weights(),
    #              'vocab': encoder.get_vocabulary()}
    #             , open(str(Config.MODEL_PATH / 'text_encoder.pkl'), "wb"))


main()
