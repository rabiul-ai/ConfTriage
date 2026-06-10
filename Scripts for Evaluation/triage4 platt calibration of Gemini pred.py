"""
Date: June 3, 2026, Md Rabiul Islam, ECEN, TAMU
Task: Gemini prediction probability is calibrated via Platt scaling
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

# =====================================================
# CONFIG
# =====================================================

input_file = r"C:\Rabiul\1. PhD Research\13. Summer 2026\3. Calibration Research\Working\Gemini and Certain Net predictions 955.xlsx"

output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage4 platt calibration of Gemini pred"
calibrated_output = os.path.join(output_folder, "Gemini Platt Calibrated 955.xlsx")
param_A_B = os.path.join(output_folder, "Platt_Parameters.txt")


# LOAD DATA
df = pd.read_excel(input_file)
nodule_ids = df["nodule ID"].values
y = df["true_label"].values
gemini_prob = df["gemini_pred_prob"].values

# =====================================================
# CREATE EXACT SAME FOLDS
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=63
)

folds = list(kf.split(nodule_ids, y))
platt_prob_all = np.zeros(len(df))
fold_info = []

eps = 1e-15

# =====================================================
# OUTER LOOP

for test_fold_idx in range(5):

    # ----------------------------------------
    # Test Fold

    test_idx = folds[test_fold_idx][1]

    # ----------------------------------------
    # Calibration Fold
    #
    # Example:
    # Fold1 test -> Fold2 calibration
    # Fold2 test -> Fold3 calibration
    # ...
    # Fold5 test -> Fold1 calibration
    # ----------------------------------------

    cal_fold_idx = (test_fold_idx + 1) % 5

    cal_idx = folds[cal_fold_idx][1]

    # =================================================
    # FIT PLATT SCALING
    # =================================================

    p_cal = gemini_prob[cal_idx]

    p_cal = np.clip(
        p_cal,
        eps,
        1 - eps
    )

    # probability -> logit

    logits_cal = np.log(
        p_cal / (1 - p_cal)
    )

    y_cal = y[cal_idx]

    # Logistic Regression
    # P(y=1)=sigmoid(A*logit+B)

    lr = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000
    )

    lr.fit(
        logits_cal.reshape(-1, 1),
        y_cal
    )

    A = lr.coef_[0][0]
    B = lr.intercept_[0]

    # =================================================
    # APPLY TO TEST FOLD
    # =================================================

    p_test = gemini_prob[test_idx]

    p_test = np.clip(
        p_test,
        eps,
        1 - eps
    )

    logits_test = np.log(
        p_test / (1 - p_test)
    )

    z = A * logits_test + B

    p_test_calibrated = (
        1 / (1 + np.exp(-z))
    )

    platt_prob_all[test_idx] = p_test_calibrated

    fold_info.append({
        "test_fold": test_fold_idx + 1,
        "calibration_fold": cal_fold_idx + 1,
        "A": A,
        "B": B
    })

# =====================================================
# SAVE CALIBRATED EXCEL

df["gemini_prob_platt_calibrated"] = platt_prob_all

df.to_excel(
    calibrated_output,
    index=False
)

# =====================================================
# SAVE A/B PARAMETERS

with open(param_A_B, "w") as f:

    f.write("Fold-wise Platt Scaling Parameters\n")
    f.write("=" * 60 + "\n\n")

    for item in fold_info:

        f.write(f"Test Fold: {item['test_fold']}\n")
        f.write(f"Calibration Fold: {item['calibration_fold']}\n")
        f.write(f"A = {item['A']:.8f}\n")
        f.write(f"B = {item['B']:.8f}\n")
        f.write("-" * 50 + "\n")

print("Done.")
print("Saved:", calibrated_output)
print("Saved:", param_A_B)