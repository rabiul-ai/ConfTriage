"""
June 2, 2026. Md Rabiul Islam. Texas A&M University
Task: I have radiologist annotation of 2625 nodules, just filtering for 955 nodules 
"""

import pandas as pd

input_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code inputs\triage2 radiologist annotation 955 only"
output_folder = r"C:\Rabiul\1. PhD Research\10. Summer 2025\1. Research 2025\3. Collaboration for VLMs\Nodule Classification\Codes\code outputs\triage2 radiologist annotation 955 only"

radiologist_annotations_2625 = pd.read_excel(f"{input_folder}/LIDC-IDRI radiologists annotation 2625.xlsx")
nodule_metadata_955 = pd.read_excel(f"{input_folder}/LIDC-IDRI nodule metadata 955.xlsx")

# Filter radiologist annotations for 955 nodules
radiologist_annotations_955 = radiologist_annotations_2625[radiologist_annotations_2625["nodule_global_id"].isin(nodule_metadata_955["all nodule count"])]

# Save the filtered radiologist annotations
radiologist_annotations_955.to_excel(f"{output_folder}/LIDC-IDRI radiologists annotation 955.xlsx", index=False)

# check if the number of nodules is correct
n_nodule= len(radiologist_annotations_955["nodule_global_id"].unique())
print(f"Nodule count in filtered file = {n_nodule}")
