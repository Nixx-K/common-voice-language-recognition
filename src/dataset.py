# Imports
import random
import shutil
import warnings

import matplotlib
matplotlib.use('Agg') # GUI-free
import matplotlib.pyplot as plt
from datacollective import load_dataset # from Mozilla Common Voice - license CC0-1.0
from dotenv import load_dotenv
import pandas as pd
from pathlib import Path
import librosa
import librosa.feature
import librosa.display
import numpy as np

load_dotenv() # load outrAPI KEY from .env

# Variables
LANGUAGES = ['pl', 'nl', 'pt']
SAMPLES =1000 # per language
RANDOM_SEED=27 # I like this number <3
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_AUG_DIR = DATA_DIR / "processed_augmented" # we'll be using augmentation to check if the model improves with worse quality training data
# Make data folders right here
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_AUG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# cols: ['audio_path', 'transcription', 'speaker_id', 'sentence_id', 'sentence_domain', 'up_votes', 'down_votes', 'age', 'gender', 'accents', 'variant', 'locale', 'segment', 'split']

AUDIO_DATA_COLUMN = "audio_path" # look through the data, it's there I swear

# For audio and melspec configuration
SAMPLE_RATE   = 16000
CLIP_DURATION = 3 # secs
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 512

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Dataset downloads (takes a few mins) - currently done in a loop, it can be done here, but will rewuire code changes
#polish = load_dataset("cmn27nz69015hmm0720txf781") # 4.5GB
#dutch =  load_dataset("cmn2g7nu901fmo107a1ydn0n5") #3.2GB
#portuguese = load_dataset("cmn29f4cb017bmm07pd9yd8mw") # 4.8GB

dataset_ids = {
    'pl': "cmn27nz69015hmm0720txf781",
    'nl': "cmn2g7nu901fmo107a1ydn0n5",
    'pt': "cmn29f4cb017bmm07pd9yd8mw",
}

clean_datasets = {}
for language, data_id in dataset_ids.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = load_dataset(data_id)
    df =df[['audio_path', 'locale']].copy()
    df = df.dropna(subset=[AUDIO_DATA_COLUMN]) # let's lose the ones withotu audio
    df = df[df[AUDIO_DATA_COLUMN].apply(lambda p: Path(p).exists())]

    # we need less samples, cause come on -> smallest one is Dutch with over 180k, we're limiting it to SAMPLES

    n = min(SAMPLES, len(df)) #in case df is smaller, but that's kind of a formality (it's not, unless you change SAMPLES)
    df = df.sample(n=n, random_state=RANDOM_SEED).reset_index(drop=True)
    df['language'] = language
    clean_datasets[language]= df

def preprocess(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(y)>= target:
        start = (len(y) - target)//2
        y=y[start:start+target]
    else:
        pad=target-len(y)
        y=np.pad(y, (pad//2,pad-pad //2))
    return y.astype(np.float32)


def add_noise_gauss(y:np.ndarray, snr_db: float=None) -> np.ndarray:
    if snr_db is None:
        snr_db = random.uniform(10.0,30.0) #randomized if not given
    power_of_signal = np.mean(y**2)+1e-10
    power_of_noise = power_of_signal / (10**(snr_db/10))
    noise = np.random.normal(0,np.sqrt(power_of_noise), len(y))
    return (y+noise).astype(np.float32)


def save_melspec(y:np.ndarray, out_path=Path): # we need to have visual representation of data, because sound aint gonna cut it
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig,ax = plt.subplots(figsize=(2.56,0.64),dpi=100)
    ax.axis('off')
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, ax=ax, cmap='magma')
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)



processed_records = [] # first .csv
augmented_records = [] # the next .csv

for lang in LANGUAGES:
    df = clean_datasets[lang]

    for i, row in df.iterrows():
        src  = Path(row['audio_path'])
        stem = f"{lang}_{src.stem}"

        # raw audio
        raw_dst = RAW_DIR / f"{stem}.mp3"
        if not raw_dst.exists():
            shutil.copy2(src, raw_dst)

        y = preprocess(str(src))

        # clean mel-spectrogram
        out_orig = PROCESSED_DIR / f"{stem}.png"
        if not out_orig.exists():
            save_melspec(y, out_orig)
        processed_records.append({'filename': f"{stem}.png", 'language': lang})

        # augmented mel-spectrogram
        out_aug = PROCESSED_AUG_DIR / f"{stem}_aug.png"
        if not out_aug.exists():
            save_melspec(add_noise_gauss(y), out_aug)
        augmented_records.append({'filename': f"{stem}_aug.png", 'language': lang})



pd.DataFrame(processed_records).to_csv(DATA_DIR / "labels_processed.csv", index=False)
pd.DataFrame(augmented_records).to_csv(DATA_DIR / "labels_augmented.csv", index=False)
