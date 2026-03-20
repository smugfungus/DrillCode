import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_mfcc.csv"
# ajuste se o seu CSV estiver em data/processed

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(DATA_PATH)

# ============================================================
# Target (COMEÇAR COM fail_in_3)
# ============================================================
TARGET = "fail_in_5"

# ============================================================
# Seleção automática das features MFCC
# ============================================================
FEATURES = [c for c in df.columns if c.startswith("mfcc_")]

print(f"Total de features MFCC: {len(FEATURES)}")

# ============================================================
# Limpeza de dados (ESSENCIAL)
# ============================================================

# remover linhas sem target
df = df.dropna(subset=[TARGET])

# garantir target inteiro
df[TARGET] = df[TARGET].astype(int)

# substituir inf/-inf por NaN nas features
df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)

# remover linhas inválidas nas features
df = df.dropna(subset=FEATURES)

# limitar outliers extremos (muito importante para MFCC)
for col in FEATURES:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    df[col] = df[col].clip(q01, q99)

# ============================================================
# Ordenação temporal
# ============================================================
df = df.sort_values("hole_id").reset_index(drop=True)

# ============================================================
# Montagem de X e y
# ============================================================
X = df[FEATURES].values
y = df[TARGET].values

# checagem final de sanidade
assert np.isfinite(X).all(), "X contém valores inválidos"

# ============================================================
# Split temporal (80/20)
# ============================================================
split_idx = int(0.8 * len(df))

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Amostras treino: {len(X_train)}")
print(f"Amostras teste : {len(X_test)}")

# ============================================================
# Modelo Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================================================
# Avaliação
# ============================================================
y_pred = rf.predict(X_test)

print("\n================ CONFUSION MATRIX ================\n")
print(confusion_matrix(y_test, y_pred))

print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(y_test, y_pred, digits=3))
