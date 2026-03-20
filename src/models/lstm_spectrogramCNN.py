import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# filepath: c:\Users\rapha\Desktop\DrillCode\src\models\lstm_spectrogram.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.preprocessing.spectrogram_dataset import SpectrogramDataset
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import numpy as np

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 12
BATCH_SIZE = 32
N_MELS = 64

# =========================
# MODELO CNN + LSTM
# =========================
class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )

        # após pooling: freq 64 → 16
        self.lstm = nn.LSTM(
            input_size=32 * (N_MELS // 4),
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # x: (batch, time, freq)
        x = x.unsqueeze(1)              # (batch, 1, time, freq)
        x = x.permute(0, 1, 3, 2)       # (batch, 1, freq, time)

        x = self.cnn(x)                 # (batch, ch, freq', time')

        x = x.permute(0, 3, 1, 2)       # (batch, time', ch, freq')
        x = x.flatten(2)                # (batch, time', features)

        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

# =========================
# DATASET
# =========================
ROOT = Path(__file__).resolve().parents[2] / "data" / "spectrograms"
dataset = SpectrogramDataset(ROOT)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# =========================
# CLASS WEIGHT
# =========================
labels = [dataset[i][1].item() for i in range(len(dataset))]
counts = Counter(labels)
total = sum(counts.values())

weights = [
    total / counts[0],
    total / counts[1]
]

class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
print("Class weights:", class_weights)

# =========================
# TREINO
# =========================
model = CNNLSTM().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} finalizada")

print("Treino finalizado.")

# =========================
# AVALIAÇÃO
# =========================
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        preds = torch.argmax(out, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(y.numpy())

cm = confusion_matrix(y_true, y_pred)
report = classification_report(
    y_true,
    y_pred,
    digits=3,
    zero_division=0
)

print("\n================ CONFUSION MATRIX ================")
print(cm)

print("\n================ CLASSIFICATION REPORT ================")
print(report)