import sys
from pathlib import Path
import pickle
import pyarabic.araby as ar
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.append(str(Path.absolute(self=Path('./'))))
from Code.constant import Constant

conditions = [ar.is_arabicrange]
morphs = [
    ar.strip_tashkeel,
    ar.strip_tatweel
]


# def fit_encoder(lyrics):
#     encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
#         max_tokens=vocab_size
#     )
#
#     encoder.adapt(lyrics)
#     return encoder

def fit_encoder(lyrics, vocab_size=10000):
    sequences = []
    tokens_list = []
    for text in lyrics:
        tokens = ar.tokenize(
            text,
            conditions,
            morphs
        )
        sequences.append(tokens)
        tokens_list.extend(tokens)

    word_count = Counter(tokens_list)
    vocab = [word for word, _ in word_count.most_common(vocab_size)]
    print(f'vocab = {vocab}')
    word_index = {word: i + 2 for i, word in enumerate(list(vocab))}
    index_word = {i + 2: word for i, word in enumerate(list(vocab))}
    index_word[0] = Constant.PAD
    index_word[1] = Constant.UNK
    word_index[Constant.PAD] = 0
    word_index[Constant.UNK] = 1
    max_seq_len = max([len(seq) for seq in sequences])
    encoder = {
        'vocab_size': len(vocab) + 2,
        'word_index': word_index,
        'index_word': index_word,
        'tokenizer_conditions': conditions,
        'tokenizer_morphs': morphs,
        'max_seq_len': max_seq_len
    }
    return encoder, sequences


def save_encoder(encode_path, encoder):
    with open(encode_path, 'wb') as file:
        pickle.dump(encoder, file)


def load_encoder(encode_path):
    with open(encode_path, "rb") as file:
        encoder = pickle.load(file)

    return encoder


def tokenize(texts):
    for text in texts:
        yield ar.tokenize(
            text,
            conditions,
            morphs
        )


# def load_encoder(encode_path):
#     with open(encode_path, "rb") as file:
#         from_disk = pickle.load(file)
#         config = from_disk['config']
#         encoder = tf.keras.layers.experimental.preprocessing.TextVectorization.from_config(config)
#         encoder.set_vocabulary(from_disk['vocab'])
#         encoder.set_weights(from_disk['weights'])
#     return encoder

def text_to_sequences(text_sequences, encoder):
    word_index = encoder.get('word_index')
    sequences = []
    for text_sequence in text_sequences:
        sequence = [word_index.get(Constant.UNK) if word_index.get(token) is None else word_index.get(token)
                    for token in text_sequence]
        sequences.append(sequence)
    return sequences


def pad_seqs(sequences, max_seq_len, pad_value=0.0):
    return pad_sequences(sequences, maxlen=max_seq_len, value=pad_value)
