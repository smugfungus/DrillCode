import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_MFCC = PROJECT_ROOT / "data" / "metadata" / "dataset_mfcc.csv"

# ============================================================
# Configuração
# ============================================================
TARGET = "holes_before_fail"   # vamos definir abaixo
HORIZON = "fail_in_5"          # mude para fail_in_1 / fail_in_3 se quiser

# ============================================================
# Load dataset
# ============================================================
df = pd.read_csv(CSV_MFCC)

# definir alvo contínuo
df["holes_before_fail"] = df[HORIZON]

df = df.dropna(subset=["holes_before_fail"]).reset_index(drop=True)

# selecionar apenas colunas MFCC
mfcc_cols = [c for c in df.columns if c.startswith("mfcc_")]

X = df[mfcc_cols].values
y = df["holes_before_fail"].values

print("Amostras:", X.shape[0])
print("Features:", X.shape[1])
print("y min / mean / max:", y.min(), y.mean(), y.max())

# ============================================================
# Train / Test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ============================================================
# Modelo
# ============================================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ============================================================
# Avaliação
# ============================================================
y_pred = model.predict(X_test)

mse = root_mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
corr, _ = pearsonr(y_test, y_pred)

print("\n=========== RESULTADOS REGRESSÃO ===========")
print(f"Horizonte: {HORIZON}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"Correlação (Pearson): {corr:.3f}")
