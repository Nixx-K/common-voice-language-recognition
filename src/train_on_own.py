import random
import warnings
import numpy as np
import tensorflow as tf
import librosa
import librosa.feature
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Variables matching train.py and dataset.py
SAMPLE_RATE   = 16_000
CLIP_DURATION = 3
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 512
IMG_WIDTH     = 256
IMG_HEIGHT    = 64

LANG_MAP    = {0: 'pl', 1: 'nl', 2: 'pt'}
OWN_DIR     = Path("data/own")
TMP_DIR     = Path("data/own_melspecs")   # throwaway PNGs
MODEL_PATH  = Path("models/best_model.keras")

TMP_DIR.mkdir(exist_ok=True)

#dataset.py helpers, because this file was added later and is considered additional, so dataset.py doesn't cover it
def preprocess(file_path: str) -> np.ndarray:
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    target = int(SAMPLE_RATE * CLIP_DURATION)
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start : start + target]
    else:
        pad = target - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2))
    return y.astype(np.float32)


def save_melspec(y: np.ndarray, out_path: Path) -> None:
    mel    = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(2.56, 0.64), dpi=100)
    ax.axis('off')
    librosa.display.specshow(mel_db, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             ax=ax, cmap='magma')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def load_png_for_model(png_path: Path) -> tf.Tensor:
    img = tf.io.read_file(str(png_path))
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    return tf.expand_dims(img, 0)   # (1, H, W, 3)

# Load the model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

# Predict
files = sorted(OWN_DIR.glob("*.m4a"))
if not files:
    raise FileNotFoundError(f"No .m4a files found in {OWN_DIR}") #justincase👉👈

for m4a in files:
    name = m4a.stem.lower()
    if name.startswith("pol"):
        true_lang = "pl"
    elif name.startswith("port"):
        true_lang = "pt"
    else:
        true_lang = "?"

    png_path = TMP_DIR / f"{m4a.stem}.png"
    y = preprocess(str(m4a))
    save_melspec(y, png_path)

    img   = load_png_for_model(png_path)
    probs = model.predict(img, verbose=0)[0]  #(3,)
    pred  = int(np.argmax(probs))
    pred_lang = LANG_MAP[pred]

    correct = "✅" if pred_lang == true_lang else "❌"
    print(f"{m4a.name:<20}  true: {true_lang}  pred: {pred_lang}  {correct}"
          f"   pl={probs[0]:.2f}  nl={probs[1]:.2f}  pt={probs[2]:.2f}")