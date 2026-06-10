import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

output_folder = r"code outputs\\triage11 theorem1 figure"
os.makedirs(output_folder, exist_ok=True)

input_file = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage10 theorem1 empirical results\Theorem1_results.xlsx"
df = pd.read_excel(input_file)

df = df.sort_values("coverage")

tau_target = 0.28

row = df.iloc[
    (df["tau"] - tau_target).abs().argsort()[:1]
]

cov = row["coverage"].values[0]
err = row["empirical_error"].values[0]

plt.figure(figsize=(7,4))

plt.plot(
    df["coverage"],
    df["empirical_error"],
    marker='o',
    linewidth=1.5,
    label='Empirical Error'
)

plt.plot(
    df["coverage"],
    df["theorem1_upper_bound"],
    linestyle='--',
    linewidth=1.5,
    label='Theorem 1 Certified Upper Bound'
)

plt.scatter(
    cov,
    err,
    s=120,
    marker='s',
    color='black',
    label='Selected τ=0.28'
)

ax = plt.gca()

ax.xaxis.set_major_formatter(
    PercentFormatter(1.0)
)

ax.yaxis.set_major_formatter(
    PercentFormatter(1.0)
)

plt.xlabel("LLM Decision Coverage (%)")
plt.ylabel("Error Rate (%)")

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "Theorem1_error_bound.png"), dpi=500, bbox_inches='tight')
plt.show()