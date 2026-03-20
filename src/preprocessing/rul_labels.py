import pandas as pd

METADATA_PATH = "data/metadata/segmented_metadata.csv"
JAM_COL = "jam_flag"  # <-- ajuste aqui se o nome for outro

def generate_rul():
    df = pd.read_csv(METADATA_PATH)

    if JAM_COL not in df.columns:
        raise ValueError(f"O CSV não tem a coluna '{JAM_COL}'.")

    df["RUL"] = None

    for drill_id in df["drill_id"].unique():
        subset = df[df["drill_id"] == drill_id]

        if subset[JAM_COL].sum() == 0:
            continue  # broca sem falha

        failure_idx = subset[subset[JAM_COL] == 1]["hole_idx"].min()

        for idx, row in subset.iterrows():
            df.loc[idx, "RUL"] = failure_idx - row["hole_idx"]

    df["RUL"] = df["RUL"].astype(int)

    df.to_csv(METADATA_PATH, index=False)
    print(f"[OK] RUL gerado e salvo em {METADATA_PATH}")

if __name__ == "__main__":
    generate_rul()
