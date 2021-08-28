import pandas as pd
import os
import sys
from pathlib import Path

sys.path.append(str(Path.absolute(self=Path('./'))))
from pydub import AudioSegment
from tqdm import tqdm


def convert_to_wav(files: list, verbose=True):
    if verbose:
        print('Start converting songs files to wav extension..\nPlease be patient :)')

    for file in tqdm(files):
        dot_index = file.rfind('.')

        extension = file[dot_index:]
        filename = file[:len(file) - len(extension)]

        src = str(SONGS_PATH / file)
        dist = str(SONGS_WAV_PATH / filename) + '.wav'
        if not os.path.exists(dist):
            sound = AudioSegment.from_mp3(src)
            sound.export(dist, format='wav')


AudioSegment.converter = "ffmpeg"

SONGS_PATH = Path(sys.argv[1])
SONGS_WAV_PATH = Path(sys.argv[2])
SONGS_WAV_PATH.mkdir(parents=True, exist_ok=True)

songs_files = os.listdir(str(SONGS_PATH))
convert_to_wav(songs_files)
