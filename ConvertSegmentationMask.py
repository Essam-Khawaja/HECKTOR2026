import nibabel as nib
import numpy as np
import pandas as pd

clinical_path = "/home/syedessamuddin.khawa/EssamProjects/HECKTOR2026/Data/HECKTOR_2026_training_data.csv"

clinical_df = pd.read_csv(clinical_path)
patient_ids = clinical_df["PatientID"].tolist()

rows = []

for patient_id in patient_ids:
    mask_path = f"/home/syedessamuddin.khawa/HECKTOR 2026 Training Data/{patient_id}/{patient_id}.nii.gz"

    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    mask_data = np.rint(mask_data).astype(np.int16)

    gtvp = mask_data == 1
    gtvn = mask_data == 2

    voxel_size = mask_img.header.get_zooms()[:3]
    voxel_volume_mm3 = voxel_size[0] * voxel_size[1] * voxel_size[2]

    row = {
        "PatientID": patient_id,
        "GTVp_voxels": int(gtvp.sum()),
        "GTVn_voxels": int(gtvn.sum()),
        "GTVp_volume_ml": gtvp.sum() * voxel_volume_mm3 / 1000,
        "GTVn_volume_ml": gtvn.sum() * voxel_volume_mm3 / 1000,
        "Total_tumor_volume_ml": (gtvp.sum() + gtvn.sum()) * voxel_volume_mm3 / 1000,
    }

    rows.append(row)

    print(patient_id, "done")

seg_df = pd.DataFrame(rows)

print(seg_df.head())
print(seg_df.shape)

merged_df = clinical_df.merge(seg_df, on="PatientID", how="left")

output_path = "/home/syedessamuddin.khawa/EssamProjects/HECKTOR2026/Data/HECKTOR_2026_training_data_with_segmentation_features.csv"

merged_df.to_csv(output_path, index=False)

print("Saved to:", output_path)
print(merged_df.head())
print(merged_df.shape)