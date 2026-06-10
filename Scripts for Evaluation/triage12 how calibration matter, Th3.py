"""
June 09, 2026. Md Rabiul Islam, ECEN, TAMU. 
Task: Checking the results before and after Platt Calibration.
"""


import pandas as pd
import numpy as np
import os

from sklearn.metrics import f1_score, brier_score_loss
from sklearn.calibration import calibration_curve

input_folder = r"C:\Rabiul\1. PhD Research\13. Summer 2026\3. Calibration Research\Working"
output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage12 how calibration matter, Th3"

# --------------------------------------------------
# Load data
# --------------------------------------------------
input_file = os.path.join(input_folder, "LIDC-IDRI FINAL metadata 955.xlsx")
df = pd.read_excel(input_file)

tau = 0.28

y_true = df["true_label"].values

# --------------------------------------------------
# ECE function
# --------------------------------------------------

def compute_ece(y_true, y_prob, n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0

    for i in range(n_bins):

        mask = bin_ids == i

        if np.sum(mask) == 0:
            continue

        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])

        ece += np.abs(acc - conf) * np.sum(mask) / len(y_true)

    return ece

# --------------------------------------------------
# Calibration metrics
# --------------------------------------------------

raw_prob = df["gemini_pred_prob"].values
cal_prob = df["gemini_prob_platt_calibrated"].values

ece_raw = compute_ece(y_true, raw_prob)
ece_cal = compute_ece(y_true, cal_prob)

brier_raw = brier_score_loss(y_true, raw_prob)
brier_cal = brier_score_loss(y_true, cal_prob)

# --------------------------------------------------
# ConfTriage with RAW confidence
# --------------------------------------------------

raw_defer = np.abs(raw_prob - 0.5) < tau

raw_final_pred = np.where(
    raw_defer,
    df["certain_net_pred_label"],
    df["gemini_pred_label"]
)

coverage_raw = np.mean(~raw_defer)

f1_raw = f1_score(y_true, raw_final_pred)

# --------------------------------------------------
# ConfTriage with CALIBRATED confidence
# --------------------------------------------------

cal_defer = np.abs(cal_prob - 0.5) < tau

cal_final_pred = np.where(
    cal_defer,
    df["certain_net_pred_label"],
    df["gemini_pred_label"]
)

coverage_cal = np.mean(~cal_defer)

f1_cal = f1_score(y_true, cal_final_pred)

# --------------------------------------------------
# Results table
# --------------------------------------------------

results = pd.DataFrame({
    "Probability Source": [
        "Raw verbalized confidence",
        "Platt-scaled confidence"
    ],
    "ECE": [ece_raw, ece_cal],
    "Brier": [brier_raw, brier_cal],
    "Coverage": [coverage_raw, coverage_cal],
    "F1": [f1_raw, f1_cal]
})

print("\n===================================================")
print("Calibration and ConfTriage Performance Comparison")
print("===================================================\n")
print(results.round(4))

# --------------------------------------------------
# Save report
# --------------------------------------------------

output_file = os.path.join(output_folder, "Calibration_Comparison_Report.txt")

with open(output_file, "w") as f:

    f.write("Calibration and ConfTriage Performance Comparison\n")
    f.write("=================================================\n\n")

    f.write(results.round(4).to_string(index=False))

    f.write("\n\n")
    f.write("Interpretation\n")
    f.write("--------------\n")
    f.write(f"ECE Improvement   : {ece_raw - ece_cal:.4f}\n")
    f.write(f"Brier Improvement : {brier_raw - brier_cal:.4f}\n")
    f.write(f"Coverage Change   : {coverage_cal - coverage_raw:.4f}\n")
    f.write(f"F1 Improvement    : {f1_cal - f1_raw:.4f}\n")

    f.write("\n\n")
    f.write("Selected operating threshold (tau) = 0.28\n")

print(f"\nReport saved to:\n{output_file}")