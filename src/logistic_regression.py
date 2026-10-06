"""
Task 4: Classification with Logistic Regression
AI & ML Internship

Binary classification using the Breast Cancer Wisconsin dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    accuracy_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer_wisconsin.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target", "target_name"])
y = df["target"]

# Train/test split with stratification to preserve class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Standardize features using training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model.
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

# Probability of the positive class.
y_prob = model.predict_proba(X_test_scaled)[:, 1]

def evaluate_threshold(threshold):
    predictions = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

default_result = evaluate_threshold(0.50)

# Find a threshold that gives recall >= 0.95 while maximizing precision.
thresholds = np.arange(0.10, 0.91, 0.01)
candidates = [evaluate_threshold(t) for t in thresholds if evaluate_threshold(t)["recall"] >= 0.95]
tuned_result = max(candidates, key=lambda r: r["precision"]) if candidates else default_result

roc_auc = roc_auc_score(y_test, y_prob)
report = classification_report(y_test, (y_prob >= 0.50).astype(int))

# Save metrics.
metrics_text = f"""Task 4 - Logistic Regression Classification

Dataset: Breast Cancer Wisconsin (Diagnostic)
Train samples: {len(X_train)}
Test samples: {len(X_test)}
Number of features: {X.shape[1]}

Default threshold: 0.50
Accuracy: {default_result['accuracy']:.4f}
Precision: {default_result['precision']:.4f}
Recall: {default_result['recall']:.4f}
ROC-AUC: {roc_auc:.4f}

Confusion Matrix at threshold 0.50:
{default_result['confusion_matrix']}

Tuned threshold: {tuned_result['threshold']:.2f}
Tuned accuracy: {tuned_result['accuracy']:.4f}
Tuned precision: {tuned_result['precision']:.4f}
Tuned recall: {tuned_result['recall']:.4f}

Classification Report at threshold 0.50:
{report}
"""
(OUT / "RESULTS.md").write_text(metrics_text, encoding="utf-8")

# Confusion matrix plot.
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(default_result["confusion_matrix"]).plot(ax=ax)
ax.set_title("Confusion Matrix - Logistic Regression")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close(fig)

# ROC curve.
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "roc_curve.png", dpi=150)
plt.close(fig)

# Sigmoid curve.
z = np.linspace(-8, 8, 400)
sigmoid = 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(z, sigmoid)
ax.axhline(0.5, linestyle="--", linewidth=1)
ax.axvline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Linear score (z)")
ax.set_ylabel("Probability")
ax.set_title("Sigmoid Function")
fig.tight_layout()
fig.savefig(OUT / "sigmoid_curve.png", dpi=150)
plt.close(fig)

# Threshold comparison.
threshold_df = pd.DataFrame([
    {"threshold": r["threshold"], "accuracy": r["accuracy"], "precision": r["precision"], "recall": r["recall"]}
    for r in [evaluate_threshold(float(t)) for t in np.arange(0.10, 0.91, 0.05)]
])
threshold_df.to_csv(OUT / "threshold_comparison.csv", index=False)

print(metrics_text)
