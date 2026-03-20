import sys
from pathlib import Path
import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# IMPORT PROJECT
# -----------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.supervised_features import extract_supervised_features

# -----------------------------
# CONFIG
# -----------------------------
METADATA_PATH = "data/metadata/dataset_supervisionado.csv"

TARGET = "fail_in_5"   # altere aqui: fail_in_1 / fail_in_3 / fail_in_5

TEST_DRILLS = 5
SEED = 42

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(METADATA_PATH)
df["output_path"] = df["output_path"].str.replace("\\", "/", regex=False)
df = df.dropna(subset=[TARGET])

# -----------------------------
# REMOVER DRILLS INDESEJADOS
# -----------------------------
EXCLUDE_DRILLS = [1, 2, 3]

df = df[~df["drill_id"].isin(EXCLUDE_DRILLS)]

print("\nDrills removidos:", EXCLUDE_DRILLS)
print("Drills restantes:", df["drill_id"].unique())

drills = df["drill_id"].unique()

random.seed(SEED)
test_drills = random.sample(list(drills), TEST_DRILLS)
train_drills = [d for d in drills if d not in test_drills]

print("\nDrills treino:", train_drills)
print("Drills teste:", test_drills)

train_df = df[df["drill_id"].isin(train_drills)]
test_df = df[df["drill_id"].isin(test_drills)]

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
X_train, y_train = [], []
X_test, y_test = [], []
hole_idx_test = []

print("\nExtraindo features TREINO")
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    path = row["output_path"]
    if not os.path.exists(path):
        continue

    feats = extract_supervised_features(path)
    if feats is not None:
        X_train.append(feats)
        y_train.append(row[TARGET])

print("\nExtraindo features TESTE")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    path = row["output_path"]
    if not os.path.exists(path):
        continue

    feats = extract_supervised_features(path)
    if feats is not None:
        X_test.append(feats)
        y_test.append(row[TARGET])
        hole_idx_test.append(row["hole_idx"])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)
hole_idx_test = np.array(hole_idx_test)

print("\nSamples após extração")
print("Train:", len(X_train))
print("Test:", len(X_test))

# -----------------------------
# GRID SEARCH
# -----------------------------
print("\nIniciando GridSearch...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nMelhores parâmetros:")
print(grid.best_params_)

model = grid.best_estimator_

# -----------------------------
# PREDIÇÃO
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# -----------------------------
# MÉTRICAS
# -----------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print("Accuracy:", round(acc,4))
print("Precision:", round(prec,4))
print("Recall:", round(rec,4))
print("F1:", round(f1,4))
print("AUC:", round(auc,4))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
print("\nConfusion Matrix")
print(cm)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix ({TARGET})")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/confusion_matrix_{TARGET}.png")
plt.close()

# -----------------------------
# ROC CURVE
# -----------------------------
fpr,tpr,_ = roc_curve(y_test,y_prob)

plt.figure(figsize=(6,5))
plt.plot(fpr,tpr,label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve ({TARGET})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/roc_curve_{TARGET}.png")
plt.close()

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,5))
plt.title(f"Feature Importance ({TARGET})")
plt.bar(range(len(importances)),importances[indices])
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/feature_importance_{TARGET}.png")
plt.close()

# -----------------------------
# PREDICTION VS TIME
# -----------------------------
order = np.argsort(hole_idx_test)

hole_idx_sorted = hole_idx_test[order]
prob_sorted = y_prob[order]
true_sorted = y_test[order]

plt.figure(figsize=(10,5))

plt.plot(
    hole_idx_sorted,
    prob_sorted,
    label="Predicted Probability"
)

plt.scatter(
    hole_idx_sorted,
    true_sorted,
    color="red",
    label="Actual Failure"
)

plt.xlabel("Hole Index (Tool Life)")
plt.ylabel("Failure Probability")
plt.title(f"Prediction vs Time ({TARGET})")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig(f"{RESULT_DIR}/prediction_vs_time_{TARGET}.png")
plt.close()

print("\nGráficos salvos em:", RESULT_DIR)