from pathlib import Path


class Config:
    MODEL_PATH = Path('Modeling')
    PLOTS_PATH = Path('Plots')
    METRICS_PATH = Path('Metrics')
    PARAMS_PATH = Path('params.yaml')
    HAPPY_DATA = Path('Data/arabic/happy/prepared_data')
    SAD_DATA = Path('Data/arabic/sad/prepared_data')
    LOUD_DATA = Path('Data/arabic/loud/prepared_data')
