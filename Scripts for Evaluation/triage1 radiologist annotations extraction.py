# Date: May 9, 2026
# Author: Md Rabiul Islam, PhD Student, ECEN, TAMU
# Task: Export per-nodule annotations from each radiologist to an Excel file.

import os

import numpy as np
import pandas as pd
import pylidc as pl
from tqdm import tqdm

# Compatibility for pylidc on newer NumPy versions (np.int removed)
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]


"""========================== Part A: Directories =========================="""
# NOTE:
# - `pylidc` reads scans from its database; we use `input_data_path` only to obtain patient IDs.
# - No images/arrays are saved here; only an Excel file is produced.
input_data_path = r"C:\Rabiul\1. PhD Research\7. Summer 2024\Coding\Dataset\LIDC-IDRI"

code_outputs = "code outputs/triage1 radiologist annotations extraction"
os.makedirs(code_outputs, exist_ok=True)
output_excel_path = os.path.join(code_outputs, "LIDC-IDRI radiologist annotations per nodule.xlsx")
cache_long_csv = os.path.join(code_outputs, "cache_long_per_radiologist.csv")


"""========================== Part B: Extraction ==========================="""
patient_ids = sorted(os.listdir(input_data_path))

if os.path.exists(cache_long_csv):
    df_long = pd.read_csv(cache_long_csv)
else:
    rows_long = []
    nodule_global_id = 0

    for pid in tqdm(patient_ids, desc="Patients"):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        if scan is None:
            continue

        nodules_annotation = scan.cluster_annotations()

        for nodule_idx, a_nodule in enumerate(nodules_annotation, start=1):
            nodule_global_id += 1

            for radiologist_idx, ann in enumerate(a_nodule, start=1):
                try:
                    mask_shape = ann.boolean_mask().shape
                except Exception:
                    mask_shape = None

                centroid = getattr(ann, "centroid", None)
                if centroid is not None:
                    centroid = tuple(int(round(x)) for x in centroid)

                rows_long.append(
                    {
                        "nodule_global_id": nodule_global_id,
                        "patient_id": pid,
                        "nodule_index_in_patient": nodule_idx,
                        "radiologist_index_in_nodule": radiologist_idx,
                        "diameter_mm": getattr(ann, "diameter", None),
                        "centroid_xyz": centroid,
                        "mask_shape_xyz": mask_shape,
                        "subtlety": getattr(ann, "subtlety", None),
                        "internalStructure": getattr(ann, "internalStructure", None),
                        "calcification": getattr(ann, "calcification", None),
                        "sphericity": getattr(ann, "sphericity", None),
                        "margin": getattr(ann, "margin", None),
                        "lobulation": getattr(ann, "lobulation", None),
                        "spiculation": getattr(ann, "spiculation", None),
                        "texture": getattr(ann, "texture", None),
                        "malignancy": getattr(ann, "malignancy", None),
                    }
                )

    df_long = pd.DataFrame(rows_long)
    df_long.to_csv(cache_long_csv, index=False)

"""==================== Part C: Create a wide (per-nodule) sheet ===================="""
wide_fields = [
    "diameter_mm",
    "centroid_xyz",
    "mask_shape_xyz",
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]

if not df_long.empty:
    base_cols = ["nodule_global_id", "patient_id", "nodule_index_in_patient"]

    # Build wide rows without pivot (avoids huge temporary allocations).
    # We only keep the first 4 radiologists per nodule (LIDC is typically <=4).
    df_tmp = df_long[base_cols + ["radiologist_index_in_nodule"] + wide_fields].copy()
    df_tmp["radiologist_index_in_nodule"] = pd.to_numeric(df_tmp["radiologist_index_in_nodule"], errors="coerce")

    wide_rows = []
    for (nid, pid, nidx), g in df_tmp.groupby(base_cols, sort=False):
        g = g.sort_values("radiologist_index_in_nodule")
        row = {"nodule_global_id": nid, "patient_id": pid, "nodule_index_in_patient": nidx}

        for j, (_, r) in enumerate(g.head(4).iterrows(), start=1):
            for f in wide_fields:
                row[f"{f}_R{j}"] = r.get(f)

        # If some clusters have >4 annotations, record how many existed.
        row["num_annotations_in_cluster"] = int(len(g))
        wide_rows.append(row)

    df_nodule_wide = pd.DataFrame(wide_rows)
else:
    df_nodule_wide = pd.DataFrame(columns=["nodule_global_id", "patient_id", "nodule_index_in_patient"])


"""========================== Part D: Save Excel ==========================="""
with pd.ExcelWriter(output_excel_path) as writer:
    df_long.to_excel(writer, sheet_name="long_per_radiologist", index=False)
    df_nodule_wide.to_excel(writer, sheet_name="wide_per_nodule", index=False)

output_folder = os.path.dirname(output_excel_path)
os.makedirs(output_folder, exist_ok=True)
print(f"Saved inside folder: {output_folder}")