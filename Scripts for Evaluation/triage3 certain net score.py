"""
June 2, 2026. Md Rabiul Islam. Texas A&M University
Task: I have Certain Net Score in pickle file, i am making just excel 
"""

import os
import pickle as pkl
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Loading certain-net scores from pickle file
filepath = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\uq_c5 threshold vs performance img 955\Image_score_Eff_955.pkl"
certain_net_pkl = pkl.load(open(filepath, 'rb'))
score_true = certain_net_pkl['y_true_all']
score_pred = certain_net_pkl['y_pred_all']
score_proba = certain_net_pkl['y_proba_all']

# Need the nodule id 
df = pd.read_excel("LIDC-IDRI nodule metadata.xlsx")
df = df[df['malignancy'] != 3]
# if n_nodule == '955':
df = df[df['nodule selected'] != 'No'] # 955_samples
df['label'] = df['malignancy'].apply(lambda x: 1 if x >= 4 else 0)
# Prepare labels
y = df['label'].values
nodule_ids = df['nodule ID'].to_numpy()
nodule_ids = [str(i) for i in nodule_ids]

# Same 5-fold split as uq_c5 (scores appended per fold in test order)
n_fold, RANDOM_STATE_RABIUL = 5, 63
kf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE_RABIUL)
folds = list(kf.split(nodule_ids, y))

nodule_ids_aligned = []
for _, test_idx in folds:
    nodule_ids_aligned.extend(nodule_ids[i] for i in test_idx)

assert len(nodule_ids_aligned) == len(score_true), (
    f"nodule/score length mismatch: {len(nodule_ids_aligned)} vs {len(score_true)}"
)

output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage3 certain net score"
os.makedirs(output_folder, exist_ok=True)
certain_net_score_df = pd.DataFrame({
    "nodule_id": [int(i) for i in nodule_ids_aligned],  # Store as number
    "true_class": score_true,
    "predicted_class": score_pred,
    "predicted_prob": score_proba,
})
# Sort by nodule_id numerically
certain_net_score_df = certain_net_score_df.sort_values(by="nodule_id").reset_index(drop=True)
certain_net_score_df.to_excel(f"{output_folder}/Certain_Net prediction with image 955.xlsx", index=False)

