import os
import pandas as pd
import numpy as np

input_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code inputs\triage7 all probability file making"
output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage7 all probability file making"

input_file_1 = os.path.join(input_folder, "LIDC-IDRI nodule metadata 955.xlsx")
nodule_metadata = pd.read_excel(input_file_1)   
input_file_2 = os.path.join(input_folder, "Gemini Platt Calibrated 955.xlsx")
gemini_prob = pd.read_excel(input_file_2)

# Making a files containing all probabilities ______________________
# we found tau = 0.28, so when LLM probability - 0.5 >= 0.28, then LLM prediction is confident, otherwise certain net prediction is used
gemini_last_6 = gemini_prob.iloc[:, -6:] # Select the last 6 columns from gemini_prob
updated_df = pd.concat([nodule_metadata, gemini_last_6], axis=1) # Concatenate nodule_metadata with the last 6 columns of gemini_prob

# Checking tau and making ConfTriage probabilities ______________________
tau = 0.28

updated_df["conftriage_prob"] = np.where(
    np.abs(updated_df["gemini_prob_platt_calibrated"] - 0.5) < tau,
    updated_df["certain_net_pred_prob"],
    updated_df["gemini_prob_platt_calibrated"]
)

# Save the updated dataframe to an excel file
updated_df.to_excel(os.path.join(output_folder, "LIDC-IDRI FINAL metadata 955.xlsx"), index=False)