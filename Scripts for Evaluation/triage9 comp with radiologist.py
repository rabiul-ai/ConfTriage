"""
June 4, 2026. Md Rabiul Islam, ECEN, TAMU.
Simple version: LIDC readers vs LOO consensus + ConfTriage / DL backstop.
Same analysis as triage8, but beginner-friendly code.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# -------------------- folders and settings --------------------
input_folder = (
    r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs"
    r"\Nodule Classification\Codes\code inputs\triage9 comp with radiologist"
)
output_folder = (
    r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs"
    r"\Nodule Classification\Codes\code outputs\triage9 comp with radiologist"
)
os.makedirs(output_folder, exist_ok=True)

max_readers = 4
tau_list = [0.30, 0.50, 0.70]

# -------------------- load data (already 955 nodules) --------------------
meta = pd.read_excel(os.path.join(input_folder, "LIDC-IDRI FINAL metadata 955.xlsx"))
rad = pd.read_excel(os.path.join(input_folder, "LIDC-IDRI radiologists annotation 955.xlsx"))

# model probabilities per nodule
model_probs = meta[["nodule ID", "certain_net_pred_prob", "gemini_prob_platt_calibrated","conftriage_prob"]].copy()
model_probs = model_probs.rename(columns={"nodule ID": "nodule_global_id"}) 
model_probs = model_probs.drop_duplicates(subset=["nodule_global_id"])

# keep max 4 readers per nodule
rad = rad.sort_values(["nodule_global_id", "radiologist_index_in_nodule"])
rad = rad.groupby("nodule_global_id", as_index=False).head(max_readers)

# remove ambiguous score 3; >3 malignant, <3 benign
rad = rad[rad["malignancy"] != 3].copy()
rad["reader_binary"] = (rad["malignancy"] > 3).astype(int)

# -------------------- leave-one-out (LOO) consensus --------------------
loo_rows = []

for nid, group in rad.groupby("nodule_global_id"):
    for _, row in group.iterrows():
        others = group[group["radiologist_index_in_nodule"] != row["radiologist_index_in_nodule"]]
        if len(others) == 0:
            continue

        med_score = others["malignancy"].median()
        if med_score == 3:
            continue

        consensus_binary = 1 if med_score > 3 else 0
        loo_rows.append(
            {
                "nodule_global_id": nid,
                "radiologist_index_in_nodule": row["radiologist_index_in_nodule"],
                "malignancy": row["malignancy"],
                "reader_binary": row["reader_binary"],
                "consensus_binary_loo": consensus_binary,
            }
        )

df_loo = pd.DataFrame(loo_rows)
df_loo = df_loo.merge(model_probs, on="nodule_global_id", how="left")
df_loo.to_excel(os.path.join(output_folder, "df_loo.xlsx"), index=False)


# -------------------- helper numbers used in loops --------------------
reader_ids = sorted(df_loo["radiologist_index_in_nodule"].unique())
comparison_table = []

# -------------------- each reader vs LOO consensus --------------------
for rid in reader_ids:
    sub = df_loo[df_loo["radiologist_index_in_nodule"] == rid]

    y_true = sub["consensus_binary_loo"].values
    y_pred = sub["reader_binary"].values
    y_score = sub["malignancy"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc_val = roc_auc_score(y_true, y_score)

    # Cohen's kappa (reader vs LOO consensus)
    p0 = np.mean(y_pred == y_true)
    p_yes1 = np.mean(y_pred == 1)
    p_yes2 = np.mean(y_true == 1)
    pe = p_yes1 * p_yes2 + (1 - p_yes1) * (1 - p_yes2)
    kappa = (p0 - pe) / (1 - pe)

    comparison_table.append(
        {
            "system": "R" + str(int(rid)),
            "threshold": "",
            "n": len(sub),
            "sensitivity": sens,
            "specificity": spec,
            "accuracy": acc,
            "auc": auc_val,
            "kappa": kappa,
        }
    )

# -------------------- ConfTriage at 3 thresholds --------------------
y_true_all = df_loo["consensus_binary_loo"].values
scores_ct = df_loo["conftriage_prob"].values
auc_ct = roc_auc_score(y_true_all, scores_ct)

for tau in tau_list:
    y_pred_ct = (scores_ct >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_ct, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)

    p0 = np.mean(y_pred_ct == y_true_all)
    p_yes1 = np.mean(y_pred_ct == 1)
    p_yes2 = np.mean(y_true_all == 1)
    pe = p_yes1 * p_yes2 + (1 - p_yes1) * (1 - p_yes2)
    kappa = (p0 - pe) / (1 - pe)

    comparison_table.append(
        {
            "system": "ConfTriage",
            "threshold": tau,
            "n": len(df_loo),
            "sensitivity": sens,
            "specificity": spec,
            "accuracy": acc,
            "auc": auc_ct,
            "kappa": kappa,
        }
    )

# -------------------- DL backstop (AUC only) --------------------
scores_dl = df_loo["certain_net_pred_prob"].values
auc_dl = roc_auc_score(y_true_all, scores_dl)
comparison_table.append(
    {
        "system": "DL_Backstop",
        "threshold": "",
        "n": len(df_loo),
        "sensitivity": "",
        "specificity": "",
        "accuracy": "",
        "auc": auc_dl,
        "kappa": "",
    }
)

# -------------------- Gemini Prob (AUC only) --------------------
scores_gemini = df_loo["gemini_prob_platt_calibrated"].values
auc_gemini = roc_auc_score(y_true_all, scores_gemini)
comparison_table.append(
    {
        "system": "Gemini_Prob",
        "threshold": "",
        "n": len(df_loo),
        "sensitivity": "",
        "specificity": "",
        "accuracy": "",
        "auc": auc_gemini,
        "kappa": "",
    }
)

df_table = pd.DataFrame(comparison_table)
df_table.to_csv(os.path.join(output_folder, "reader_comparison_table.csv"), index=False)
df_table.to_excel(os.path.join(output_folder, "reader_comparison_table.xlsx"), index=False)

# -------------------- pairwise Cohen's kappa between readers --------------------
wide = rad.pivot_table(
    index="nodule_global_id",
    columns="radiologist_index_in_nodule",
    values="reader_binary",
    aggfunc="first",
)
reader_names = ["R" + str(int(c)) for c in wide.columns]
wide.columns = reader_names

pairwise_kappa = []
for i in range(len(reader_names)):
    for j in range(i + 1, len(reader_names)):
        r1 = reader_names[i]
        r2 = reader_names[j]
        sub = wide[[r1, r2]].dropna()
        y1 = sub[r1].astype(int).values
        y2 = sub[r2].astype(int).values
        p0 = np.mean(y1 == y2)
        p_yes1 = np.mean(y1 == 1)
        p_yes2 = np.mean(y2 == 1)
        pe = p_yes1 * p_yes2 + (1 - p_yes1) * (1 - p_yes2)
        k = (p0 - pe) / (1 - pe)
        pairwise_kappa.append({"reader_a": r1, "reader_b": r2, "kappa": k, "n": len(sub)})

pd.DataFrame(pairwise_kappa).to_csv(
    os.path.join(output_folder, "pairwise_cohen_kappa.csv"), index=False
)

# -------------------- ROC overlay figure --------------------
fpr_ct, tpr_ct, _ = roc_curve(y_true_all, scores_ct)
fpr_dl, tpr_dl, _ = roc_curve(y_true_all, scores_dl)
fpr_gemini, tpr_gemini, _ = roc_curve(y_true_all, scores_gemini)


plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.plot(fpr_ct, tpr_ct, linewidth=1.5, label="ConfTriage (AUC=%.3f)" % auc_ct)
plt.plot(fpr_dl, tpr_dl, linewidth=1.5,  label="Certain-Net (AUC=%.3f)" % auc_dl)
plt.plot(fpr_gemini, tpr_gemini, linewidth=1.5,  label="LLM (AUC=%.3f)" % auc_gemini)

# reader operating points
reader_points = []
for rid in reader_ids:
    sub = df_loo[df_loo["radiologist_index_in_nodule"] == rid]
    y_true = sub["consensus_binary_loo"].values
    y_pred = sub["reader_binary"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    fpr_point = fp / (fp + tn)
    reader_points.append((fpr_point, sens))
    # Use four different colors for four points (assuming reader_ids is of length 4)
    reader_colors = ['pink', 'violet', 'tab:red', 'olive']
    color = reader_colors[int(rid) % 4] if len(reader_colors) >= 4 else None
    plt.scatter(fpr_point, sens, s=30, color=color)
    plt.text(fpr_point + 0.01, sens - 0.02, "R" + str(int(rid)), fontsize=8)

# simple convex hull line for reader envelope
if len(reader_points) >= 3:
    pts = sorted(set(reader_points))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    hx = [p[0] for p in hull] + [hull[0][0]]
    hy = [p[1] for p in hull] + [hull[0][1]]
    plt.plot(hx, hy, "k-", linewidth=1, label="LOO reader operating points")

# ConfTriage points at tau values
for tau in tau_list:
    y_pred_ct = (scores_ct >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_ct, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    fpr_point = fp / (fp + tn)
    plt.scatter(fpr_point, sens, s=35, marker="d", color="black")  # diamond marker in specific color
    plt.text(fpr_point - 0.075, sens + 0.01, "t =%.1f" % tau, fontsize=8)

plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
# plt.title("Reader comparison: ROC overlay vs LOO consensus")
plt.xlim(-0.02, 1.02)
plt.ylim(-0.02, 1.02)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3) # , linestyle='--'
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "reader_comparison_roc_overlay.png"), dpi=300)
plt.close()

print("Done. Saved outputs to:", output_folder)
