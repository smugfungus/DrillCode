import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# =============================
# CONFIG
# =============================

SEED = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# =============================
# PATHS
# =============================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "spectrograms"
META_PATH = DATA_DIR / "metadata.csv"

# =============================
# LOAD DATA
# =============================

df = pd.read_csv(META_PATH)

print("Total samples:", len(df))
print("Total holes:", df["hole_id"].nunique())

# =============================
# SPLIT POR HOLE_ID
# =============================

holes = df["hole_id"].unique()

# Primeiro separa treino (70%) e temp (30%)
holes_train, holes_temp = train_test_split(
    holes,
    train_size=TRAIN_SIZE,
    random_state=SEED,
    shuffle=True
)

# Depois divide temp em val (15%) e test (15%)
relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)

holes_val, holes_test = train_test_split(
    holes_temp,
    test_size=relative_test_size,
    random_state=SEED,
    shuffle=True
)

# =============================
# FILTRAR DATAFRAME
# =============================

train_df = df[df["hole_id"].isin(holes_train)].reset_index(drop=True)
val_df = df[df["hole_id"].isin(holes_val)].reset_index(drop=True)
test_df = df[df["hole_id"].isin(holes_test)].reset_index(drop=True)

# =============================
# SALVAR CSVs
# =============================

train_df.to_csv(DATA_DIR / "train.csv", index=False)
val_df.to_csv(DATA_DIR / "val.csv", index=False)
test_df.to_csv(DATA_DIR / "test.csv", index=False)

print("\nSplit concluído!\n")

# =============================
# FUNÇÃO PARA MOSTRAR DISTRIBUIÇÃO
# =============================

def show_distribution(name, dataframe):
    print(f"--- {name} ---")
    print("Samples:", len(dataframe))
    print("Holes:", dataframe["hole_id"].nunique())
    print(dataframe["fail_in_1"].value_counts(normalize=True))
    print()

show_distribution("TRAIN", train_df)
show_distribution("VAL", val_df)
show_distribution("TEST", test_df)