import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score



output_folder = r"code outputs\\triage10 theorem1 empirical results"
os.makedirs(output_folder, exist_ok=True)

input_gemini_calibrated= r"C:\Rabiul\1. PhD Research\13. Summer 2026\3. Calibration Research\Working\Gemini Platt Calibrated 955.xlsx"
df = pd.read_excel(input_gemini_calibrated)
y_true = df["true_label"].values

gemini_prob = df["gemini_prob_platt_calibrated"].values
certain_prob = df["certain_net_pred_prob"].values

gemini_pred = (gemini_prob >= 0.5).astype(int)
certain_pred = (certain_prob >= 0.5).astype(int)

delta = 0.05
n = len(df)

hoeffding_slack = np.sqrt(np.log(1/delta)/(2*n))

results = []

tau_values = np.arange(0.00, 0.51, 0.01)

for tau in tau_values:

    confident = np.abs(gemini_prob - 0.5) >= tau

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

    empirical_error = (final_pred != y_true).mean()

    certified_error = empirical_error + hoeffding_slack

    coverage = confident.mean()

    deferral = 1 - coverage

    results.append({
        "tau": tau,
        "coverage": coverage,
        "deferral_rate": deferral,
        "empirical_error": empirical_error,
        "theorem1_upper_bound": certified_error
    })

results_df = pd.DataFrame(results)

results_df.to_excel(
    os.path.join(output_folder, "Theorem1_results.xlsx"),
    index=False
)

print(results_df.head())