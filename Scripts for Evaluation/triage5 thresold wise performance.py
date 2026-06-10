"""
June 3, 2026. Md Rabiul Islam, ECEN, TAMU.
Task: Evaluating performance on different threshold
"""

import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

output_folder = r"code outputs\\triage5 thresold wise performance"
# os.makedirs(output_folder, exist_ok=True)

# =====================================================
# LOAD CALIBRATED FILE
# =====================================================
input_gemini_calibrated= r"C:\Rabiul\1. PhD Research\13. Summer 2026\3. Calibration Research\Working\Gemini Platt Calibrated 955.xlsx"
df = pd.read_excel(input_gemini_calibrated)
y_true = df["true_label"].values

gemini_prob = df["gemini_prob_platt_calibrated"].values
certain_prob = df["certain_net_pred_prob"].values
certain_pred = df["certain_net_pred_label"].values

# =====================================================
# TAU SWEEP
# =====================================================

results = []

tau_values = np.arange(
    0.00,
    0.51,
    0.01
)

for tau in tau_values:

    # ------------------------------------
    # Gemini confidence test
    # ------------------------------------

    confident = (
        np.abs(gemini_prob - 0.5)
        >= tau
    )

    gemini_pred = (
        gemini_prob >= 0.5
    ).astype(int)

    # ------------------------------------
    # ConfTriage final prediction
    # ------------------------------------

    final_pred = np.where(
        confident,
        gemini_pred,
        certain_pred
    )

    final_prob = np.where(
        confident,
        gemini_prob,
        certain_prob
    )

    # ------------------------------------
    # Metrics
    # ------------------------------------

    accuracy = accuracy_score(
        y_true,
        final_pred
    )

    f1 = f1_score(
        y_true,
        final_pred
    )

    sensitivity = recall_score(
        y_true,
        final_pred
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        final_pred
    ).ravel()

    specificity = tn / (tn + fp)

    auc = roc_auc_score(
        y_true,
        final_prob
    )

    coverage = confident.mean()

    deferral = 1 - coverage

    results.append({
        "tau": tau,
        "coverage": coverage,
        "deferral_rate": deferral,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc
    })

# =====================================================
# SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)

results_df.to_excel(
    os.path.join(output_folder, "ConfTriage_tau_sweep.xlsx"),
    index=False
)

print(results_df.head())
print(
    "\nSaved: ConfTriage_tau_sweep.xlsx"
)