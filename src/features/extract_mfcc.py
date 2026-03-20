import pandas as pd
import numpy as np
from pathlib import Path
import librosa
from tqdm import tqdm

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_IN = PROJECT_ROOT / "data" / "metadata" / "dataset_supervisionado.csv"
CSV_OUT = PROJECT_ROOT / "data" / "metadata" / "dataset_mfcc.csv"

# ============================================================
# Configurações MFCC
# ============================================================
N_MFCC = 20

# ============================================================
# Load metadata
# ============================================================
df = pd.read_csv(CSV_IN)

# manter apenas linhas com path válido
df = df.dropna(subset=["output_path"]).reset_index(drop=True)

# ============================================================
# Função de extração
# ============================================================
def extract_mfcc_features(wav_path, n_mfcc=20):
    try:
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        features = {}

        for i in range(n_mfcc):
            coef = mfcc[i, :]
            features[f"mfcc_{i+1}_mean"] = np.mean(coef)
            features[f"mfcc_{i+1}_std"]  = np.std(coef)
            features[f"mfcc_{i+1}_min"]  = np.min(coef)
            features[f"mfcc_{i+1}_max"]  = np.max(coef)

        return features

    except Exception as e:
        print(f"Erro em {wav_path}: {e}")
        return None

# ============================================================
# Loop principal
# ============================================================
rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    abs_path = PROJECT_ROOT / Path(row["output_path"])

    if not abs_path.exists():
        continue

    feats = extract_mfcc_features(abs_path, N_MFCC)

    if feats is None:
        continue

    base_info = {
        "hole_id": row["hole_id"],
        "fail_in_1": row.get("fail_in_1"),
        "fail_in_3": row.get("fail_in_3"),
        "fail_in_5": row.get("fail_in_5"),
    }

    rows.append({**base_info, **feats})

# ============================================================
# Salvar dataset
# ============================================================
df_mfcc = pd.DataFrame(rows)
df_mfcc.to_csv(CSV_OUT, index=False)

print(f"\nMFCC extraído com sucesso: {CSV_OUT}")
print(f"Total de amostras: {len(df_mfcc)}")
