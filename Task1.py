import nibabel as nib
import numpy as np

data_path = "/home/syedessamuddin.khawa/HECKTOR 2026 Training Data/"
patient = "CHUM-001"
base = f"{data_path}{patient}/"

ct_path = f"{base}/{patient}__CT.nii.gz"
pet_path = f"{base}/{patient}__PT.nii.gz"
mask_path = f"{base}/{patient}.nii.gz"

ct_img = nib.load(ct_path)
pet_img = nib.load(pet_path)
mask_img = nib.load(mask_path)

ct = ct_img.get_fdata()
pet = pet_img.get_fdata()
mask = mask_img.get_fdata()

print("CT shape:", ct.shape)
print("PET shape:", pet.shape)
print("Mask shape:", mask.shape)

print("CT spacing:", ct_img.header.get_zooms())
print("PET spacing:", pet_img.header.get_zooms())
print("Mask spacing:", mask_img.header.get_zooms())

print("Mask labels:", np.unique(mask))