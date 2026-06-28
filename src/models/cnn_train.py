import pandas as pd
import numpy as np
from pathlib import Path
import random
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# CONFIG
# =========================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET = "fail_in_5"

EXCLUDE_DRILLS = [1, 2, 3]
TEST_DRILLS = 5

DATA_DIR = Path("data/spectrograms")
META_PATH = DATA_DIR / "metadata.csv"

RESULT_DIR = Path("results") / TARGET
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv(META_PATH)
df = df[~df["drill_id"].isin(EXCLUDE_DRILLS)]

drills = df["drill_id"].unique()

test_drills = random.sample(list(drills), TEST_DRILLS)
train_drills = [d for d in drills if d not in test_drills]

train_df = df[df["drill_id"].isin(train_drills)]
test_df  = df[df["drill_id"].isin(test_drills)]

val_drills = random.sample(list(train_df["drill_id"].unique()), 2)

val_df = train_df[train_df["drill_id"].isin(val_drills)]
train_df = train_df[~train_df["drill_id"].isin(val_drills)]

# =========================================
# DATASET
# =========================================

class SpectrogramDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        spec = np.load(DATA_DIR / row["spec_path"])

        spec = spec / (np.max(np.abs(spec)) + 1e-6)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(row[TARGET], dtype=torch.float32)

        return spec, label, row["hole_idx"]

# =========================================
# MODEL
# =========================================

class CNN(nn.Module):
    def __init__(self, dropout=0.4):
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

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# =========================================
# LOSS
# =========================================

pos = train_df[TARGET].sum()
neg = len(train_df) - pos
pos_weight = torch.tensor(neg / pos).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# =========================================
# QUICK TRAIN (RANDOM SEARCH)
# =========================================

def quick_train_eval(model, optimizer, train_loader, val_loader, epochs=8):

    for _ in range(epochs):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(DEVICE)
            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.numpy())

    preds_bin = (np.array(all_preds) > 0.5).astype(int)
    return f1_score(all_labels, preds_bin)

# =========================================
# RANDOM SEARCH
# =========================================

print("\n=== RANDOM SEARCH ===")

search_space = {
    "lr": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32],
    "dropout": [0.2, 0.4]
}

N_TRIALS = 5

best_f1 = 0
best_config = None

for trial in range(N_TRIALS):

    config = {
        "lr": random.choice(search_space["lr"]),
        "batch_size": random.choice(search_space["batch_size"]),
        "dropout": random.choice(search_space["dropout"])
    }

    print(f"\nTrial {trial+1}: {config}")

    train_loader_tmp = DataLoader(SpectrogramDataset(train_df), batch_size=config["batch_size"], shuffle=True)
    val_loader_tmp   = DataLoader(SpectrogramDataset(val_df), batch_size=config["batch_size"])

    model_tmp = CNN(dropout=config["dropout"]).to(DEVICE)
    optimizer_tmp = torch.optim.Adam(model_tmp.parameters(), lr=config["lr"])

    f1 = quick_train_eval(model_tmp, optimizer_tmp, train_loader_tmp, val_loader_tmp)

    print(f"F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_config = config

# fallback segurança
if best_config is None:
    best_config = {"lr": 1e-3, "batch_size": 32, "dropout": 0.4}

print("\nBest config:", best_config)

# =========================================
# TREINO FINAL
# =========================================

print("\n=== TREINO FINAL ===")

train_loader = DataLoader(SpectrogramDataset(train_df), batch_size=best_config["batch_size"], shuffle=True)
val_loader   = DataLoader(SpectrogramDataset(val_df), batch_size=best_config["batch_size"])

model = CNN(dropout=best_config["dropout"]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=best_config["lr"])

best_f1 = 0
EPOCHS = 25

train_losses = []
val_losses = []

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0

    for x, y, _ in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0

    with torch.no_grad():
        for x, y, _ in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x).squeeze()
            loss = criterion(logits, y)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    preds_bin = (np.array(all_preds) > 0.5).astype(int)

    f1 = f1_score(all_labels, preds_bin)
    auc = roc_auc_score(all_labels, all_preds)

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val F1: {f1:.4f} | AUC: {auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), RESULT_DIR / "best_model.pth")

# =========================================
# TEST
# =========================================

model.load_state_dict(torch.load(RESULT_DIR / "best_model.pth"))
model.eval()

all_preds, all_labels, all_holes = [], [], []

test_loader = DataLoader(SpectrogramDataset(test_df), batch_size=best_config["batch_size"])

with torch.no_grad():
    for x, y, hole_idx in test_loader:
        x = x.to(DEVICE)

        logits = model(x).squeeze()
        probs = torch.sigmoid(logits)

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(y.numpy())
        all_holes.extend(hole_idx.numpy())

preds_bin = (np.array(all_preds) > 0.5).astype(int)

# =========================================
# MÉTRICAS
# =========================================

acc = accuracy_score(all_labels, preds_bin)
prec = precision_score(all_labels, preds_bin, zero_division=0)
rec = recall_score(all_labels, preds_bin)
f1 = f1_score(all_labels, preds_bin)
auc = roc_auc_score(all_labels, all_preds)

print("\nRESULTADOS TESTE")
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)
print("AUC:", auc)

# =========================================
# GRÁFICOS
# =========================================

# Confusion Matrix
cm = confusion_matrix(all_labels, preds_bin)
plt.figure(figsize=(6,5), dpi=300)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix ({TARGET})")
plt.tight_layout()
plt.savefig(RESULT_DIR / f"confusion_matrix_{TARGET}.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_preds)
plt.figure(figsize=(6,5), dpi=300)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({TARGET})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / f"roc_curve_{TARGET}.png")
plt.close()

# Prediction vs Time
holes = np.array(all_holes)
probs = np.array(all_preds)
labels = np.array(all_labels)

order = np.argsort(holes)

plt.figure(figsize=(10,5), dpi=300)
plt.plot(holes[order], probs[order], label="Predicted Probability")
plt.scatter(holes[order], labels[order], color="red", label="Actual Failure")
plt.xlabel("Hole Index (Tool Life)")
plt.ylabel("Failure Probability")
plt.title(f"Prediction vs Time ({TARGET})")
plt.legend()
plt.grid(alpha=0.3)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(RESULT_DIR / f"prediction_vs_time_{TARGET}.png")
plt.close()

# Loss curve
plt.figure(figsize=(6,5), dpi=300)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "loss_curve.png")
plt.close()