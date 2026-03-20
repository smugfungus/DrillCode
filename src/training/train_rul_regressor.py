import pandas as pd
import os
import re

METADATA_PATH = "data/metadata/dataset_supervisionado.csv"
FILEPATH_COL = "output_path"  # <<< AJUSTE AQUI
JAM_COL = "jam_flag"

def extract_drill_id(filepath):
    filename = os.path.basename(filepath)
    match = re.match(r"(\d+)_", filename)
    return match.group(1) if match else "unknown"

def extract_hole_idx(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r"hole(\d+)", filename)
    return int(match.group(1)) if match else None

def generate_rul():
    df = pd.read_csv(METADATA_PATH)

    if FILEPATH_COL not in df.columns:
        raise ValueError(f"CSV precisa ter a coluna '{FILEPATH_COL}'")

    # drill_id
    if "drill_id" not in df.columns:
        df["drill_id"] = df[FILEPATH_COL].apply(extract_drill_id)

    # hole_idx
    if "hole_idx" not in df.columns:
        df["hole_idx"] = df[FILEPATH_COL].apply(extract_hole_idx)

    df["hole_idx"] = pd.to_numeric(df["hole_idx"], errors="coerce")

    # jam_flag
    if JAM_COL not in df.columns:
        df[JAM_COL] = df[FILEPATH_COL].str.contains("jam", case=False).astype(int)

    df["RUL"] = -1  # censurado por padrão

    for drill_id in df["drill_id"].unique():
        subset = df[df["drill_id"] == drill_id]

        if subset[JAM_COL].sum() == 0:
            continue

        failure_idx = subset.loc[subset[JAM_COL] == 1, "hole_idx"].min()
        df.loc[subset.index, "RUL"] = failure_idx - subset["hole_idx"]

    df.to_csv(METADATA_PATH, index=False)
    print("[OK] RUL calculado corretamente.")

if __name__ == "__main__":
    generate_rul()
