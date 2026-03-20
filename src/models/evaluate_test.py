import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================
# PATHS
# =========================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "spectrograms"

TEST_CSV = DATA_DIR / "test.csv"
MODEL_PATH = PROJECT_ROOT / "best_cnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# DATASET
# =========================================

class SpectrogramDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        spec_path = DATA_DIR / row["spec_path"]

        spec = np.load(spec_path)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(row["fail_in_1"], dtype=torch.float32)

        return spec, label


test_dataset = SpectrogramDataset(TEST_CSV)
test_loader = DataLoader(test_dataset, batch_size=32)

# =========================================
# MODEL (mesma arquitetura do treino)
# =========================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =========================================
# EVALUATION
# =========================================

all_probs = []
all_labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x).squeeze()
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())

all_probs = np.array(all_probs)
all_preds = (all_probs > 0.5).astype(int)

print("\n===== TEST RESULTS =====")
print(classification_report(all_labels, all_preds))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

auc = roc_auc_score(all_labels, all_probs)
print("Test AUC:", round(auc, 4))