from pathlib import Path


class Config:
    DATA_PATH = Path('Data/PMEmo2019')
    DATA_PATH_PREPARED = DATA_PATH / Path('prepared')
    SONGS_PATH = DATA_PATH / Path('songs')
    SONGS_WAV_PATH = DATA_PATH / Path('songs_wav')
    META_DATA_PATH = str(DATA_PATH / Path('metadata.csv'))
    FEATURES_PATH = DATA_PATH / Path('features')
    FEATURES_FILE = str(FEATURES_PATH / Path('static_features.csv'))
    ANNOTATIONS_FILE = str(DATA_PATH / Path('annotations/static_annotations.csv'))
    MODEL_PATH = Path('Modeling')
    PLOTS_PATH = Path('Plots')
    METRICS_PATH = Path('Metrics')
    PARAMS_PATH = Path('params.yaml')
