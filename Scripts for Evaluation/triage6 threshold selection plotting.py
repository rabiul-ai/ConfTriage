import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import PercentFormatter

filepath = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage5 thresold wise performance"
df = pd.read_excel(os.path.join(filepath, "ConfTriage_tau_sweep.xlsx"))

output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage6 threshold selection plotting"

df = df.sort_values("coverage")

x = df["coverage"].values
y = df["f1"].values

# --------------------------------------------------
# Elbow detection
# --------------------------------------------------

p1 = np.array([x[0], y[0]])
p2 = np.array([x[-1], y[-1]])

distances = []

for i in range(len(x)):
    p = np.array([x[i], y[i]])
    dist = np.abs(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
    distances.append(dist)

elbow_idx = np.argmax(distances)

elbow_tau = df.iloc[elbow_idx]["tau"]
elbow_cov = df.iloc[elbow_idx]["coverage"]
elbow_f1 = df.iloc[elbow_idx]["f1"]

print(f"Elbow tau = {elbow_tau:.2f}")
print(f"Coverage = {elbow_cov:.4f}")
print(f"F1 = {elbow_f1:.4f}")

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(7,4))

plt.plot(x, y, color='red', marker='o', linewidth=1.25)

# plt.axvspan(x.min(), elbow_cov, color='green', alpha=0.12)

plt.axvline(elbow_cov, color='black', linestyle='--', linewidth=1)

# plt.scatter(elbow_cov, elbow_f1, color='black', s=120, marker='s', label=f'Elbow τ={elbow_tau:.2f}')






# --------------------------------------------------
# Special operating points
# --------------------------------------------------

llm_row = df.iloc[(df["tau"] - 0.00).abs().argsort()[:1]]
conf_row = df.iloc[(df["tau"] - 0.28).abs().argsort()[:1]]
dl_row = df.iloc[(df["tau"] - 0.50).abs().argsort()[:1]]

# LLM only
plt.scatter(
    llm_row["coverage"].values[0],
    llm_row["f1"].values[0],
    color='blue',
    s=120,
    marker='o'
)

plt.annotate(
    "LLM only\n(τ=0.00)",
    (llm_row["coverage"].values[0], llm_row["f1"].values[0]),
    xytext=(-50,-1),
    textcoords='offset points',
    fontsize=9
)

# ConfTriage
plt.scatter(
    conf_row["coverage"].values[0],
    conf_row["f1"].values[0],
    color='black',
    s=140,
    marker='s'
)

plt.annotate(
    "ConfTriage\n(τ=0.28)",
    (conf_row["coverage"].values[0], conf_row["f1"].values[0]),
    xytext=(20,-10),
    textcoords='offset points',
    fontsize=9
)

# Certain-Net only
plt.scatter(
    dl_row["coverage"].values[0],
    dl_row["f1"].values[0],
    color='green',
    s=120,
    marker='^'
)

plt.annotate(
    "Certain-Net only\n(τ=0.50)",
    (dl_row["coverage"].values[0], dl_row["f1"].values[0]),
    xytext=(-10,10),
    textcoords='offset points',
    fontsize=9
)



# plt.annotate(f'τ={elbow_tau:.2f}\nCoverage={elbow_cov*100:.1f}%\nF1={elbow_f1*100:.1f}%',
#              (elbow_cov, elbow_f1),
#              xytext=(10,10),
#              textcoords='offset points')

plt.text(0.52,
         y.min()+0.012,
         f'ConfTriage resolved {elbow_cov*100:.1f}% cases \n using zero-shot LLM inference alone',
         ha='center',
         color='black',
         fontsize=10)

ax = plt.gca()
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

plt.xlabel("LLM Decision Coverage (%)")
plt.ylabel("F1 Score")
# plt.title("Coverage–F1 Tradeoff for ConfTriage")

plt.grid(True, alpha=0.3)
# plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(output_folder, "ConfTriage_threshold_selection.png"), dpi=500, bbox_inches='tight')

plt.show()