import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, target_column="fail_in_1"):
        self.root_dir = Path(root_dir)
        self.meta = pd.read_csv(self.root_dir / "metadata.csv")
        if target_column not in self.meta.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in metadata.csv. "
                f"Available columns: {list(self.meta.columns)}"
            )
        self.target_column = target_column

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        spec = np.load(self.root_dir / row["spec_path"])
        spec = torch.tensor(spec).transpose(0, 1)  # (time, freq)
        label = torch.tensor(row[self.target_column], dtype=torch.float32)
        return spec, label
