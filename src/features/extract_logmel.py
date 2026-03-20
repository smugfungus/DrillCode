import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_IN = PROJECT_ROOT / "data" / "metadata" / "dataset_supervisionado.csv"
OUT_DIR = PROJECT_ROOT / "data" / "spectrograms"

OUT_DIR.mkdir(parents=True, exist_ok=True)

N_MELS = 64
N_FFT = 1024
HOP = 512
MAX_FRAMES = 128

df = pd.read_csv(CSV_IN)
df = df.dropna(subset=["output_path", "fail_in_1", "fail_in_3", "fail_in_5"]).reset_index(drop=True)

meta = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = PROJECT_ROOT / Path(row["output_path"])
    if not wav_path.exists():
        continue

    y, sr = librosa.load(wav_path, sr=None, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0
    )

    # escala fixa (importante)
    logmel = librosa.power_to_db(mel, ref=1.0)

    # pad / truncate no tempo
    if logmel.shape[1] < MAX_FRAMES:
        pad = MAX_FRAMES - logmel.shape[1]
        min_val = logmel.min()
        logmel = np.pad(
            logmel,
            ((0, 0), (0, pad)),
            mode="constant",
            constant_values=min_val
        )
    else:
        logmel = logmel[:, :MAX_FRAMES]

    out_file = OUT_DIR / f"spec_{i:06d}.npy"
    np.save(out_file, logmel.astype(np.float32))

    meta.append({
    "spec_path": out_file.name,
    "hole_id": row["hole_id"],
    "drill_id": row["drill_id"],
    "hole_idx": row["hole_idx"],
    "fail_in_1": int(row["fail_in_1"]),
    "fail_in_3": int(row["fail_in_3"]),
    "fail_in_5": int(row["fail_in_5"])
    })  

pd.DataFrame(meta).to_csv(OUT_DIR / "metadata.csv", index=False)
print("Espectrogramas extraídos com sucesso.")