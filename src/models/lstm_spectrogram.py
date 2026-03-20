import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# filepath: c:\Users\rapha\Desktop\DrillCode\src\models\lstm_spectrogram.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.preprocessing.spectrogram_dataset import SpectrogramDataset
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 32
TARGET_COLUMNS = ["fail_in_1", "fail_in_3", "fail_in_5"]

# =========================
# MODELO
# =========================
class LSTMSpec(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=64,      # n_mels
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze(-1)

# =========================
# DATASET / TREINO / AVALIAÇÃO
# =========================
ROOT = Path(__file__).resolve().parents[2] / "data" / "spectrograms"

for target_column in TARGET_COLUMNS:
    print(f"\n================ TARGET: {target_column} ================")

    dataset = SpectrogramDataset(ROOT, target_column=target_column)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # =========================
    # NORMALIZAÇÃO DO ALVO (TREINO)
    # =========================
    train_targets = []

    for x, y in train_loader:
        train_targets.extend(y.numpy().tolist())
    train_targets = torch.tensor(train_targets, dtype=torch.float32)

    y_mean = train_targets.mean().item()
    y_std = train_targets.std(unbiased=False).item()
    if y_std == 0:
        y_std = 1.0
    print(f"Target mean: {y_mean:.6f} | std: {y_std:.6f}")

    model = LSTMSpec().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = ((y - y_mean) / y_std).to(DEVICE)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} finalizada")

    print("Treino finalizado.")

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x).cpu().numpy()
            y_pred.extend(out.tolist())
            y_true.extend(y.numpy().tolist())

    # Desnormaliza para métricas em escala original
    y_pred = [p * y_std + y_mean for p in y_pred]

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n================ REGRESSION METRICS ================")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE : {mae:.6f}")
    print(f"R2  : {r2:.6f}")

    # =========================
    # BASELINE (MÉDIA DO TREINO)
    # =========================
    baseline_pred = [y_mean] * len(y_true)
    b_mse = mean_squared_error(y_true, baseline_pred)
    b_rmse = np.sqrt(b_mse)
    b_mae = mean_absolute_error(y_true, baseline_pred)
    b_r2 = r2_score(y_true, baseline_pred)

    print("\n================ BASELINE (MEAN) ================")
    print(f"MSE : {b_mse:.6f}")
    print(f"RMSE: {b_rmse:.6f}")
    print(f"MAE : {b_mae:.6f}")
    print(f"R2  : {b_r2:.6f}")
