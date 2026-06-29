import os, json, random

data_root = "/home/syedessamuddin.khawa/HECKTOR 2026 Training Data"
out_file = "/home/syedessamuddin.khawa/EssamProjects/HECKTOR2026/Task/Segmentation/config/splits_smoke.json"

valid_cases = []

for case_id in sorted(os.listdir(data_root)):
    case_dir = os.path.join(data_root, case_id)
    if not os.path.isdir(case_dir):
        continue

    ct = os.path.join(case_dir, f"{case_id}__CT.nii.gz")
    pet = os.path.join(case_dir, f"{case_id}__PT.nii.gz")
    label = os.path.join(case_dir, f"{case_id}.nii.gz")

    if os.path.exists(ct) and os.path.exists(pet) and os.path.exists(label):
        valid_cases.append(case_id)

print("Valid cases:", len(valid_cases))
print("First 10:", valid_cases[:10])

random.seed(42)
random.shuffle(valid_cases)

smoke_train = valid_cases[:8]
smoke_val = valid_cases[8:10]

splits = [
    {
        "train": smoke_train,
        "val": smoke_val
    }
]

with open(out_file, "w") as f:
    json.dump(splits, f, indent=2)

print("Wrote:", out_file)
print("Train:", smoke_train)
print("Val:", smoke_val)