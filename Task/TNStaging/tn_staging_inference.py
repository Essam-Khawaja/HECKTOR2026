import os
import argparse
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Resized
)

from tn_staging import MultiModalTNModel, HecktorTNDataset, T_STAGES, N_STAGES


img_transforms = Compose([
    LoadImaged(keys=["ct", "pet"]),
    EnsureChannelFirstd(keys=["ct", "pet"]),
    ScaleIntensityd(keys=["ct", "pet"]),
    Resized(keys=["ct", "pet"], spatial_size=(96, 96, 96)),
    ToTensord(keys=["ct", "pet"]),
])


def run_inference(csv_path, img_dir, checkpoint, scaler_file, ohe_file, output_path, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    with open(ohe_file, "rb") as f:
        ohe = pickle.load(f)

    t_encoder = LabelEncoder().fit(T_STAGES)
    n_encoder = LabelEncoder().fit(N_STAGES)

    dataset = HecktorTNDataset(
        csv_file=csv_path,
        img_dir=img_dir,
        transforms=img_transforms,
        scaler=scaler,
        ohe=ohe,
        t_encoder=t_encoder,
        n_encoder=n_encoder,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    clin_feat_dim = dataset.clinical_feats.shape[1]
    model = MultiModalTNModel(
        clin_feat_dim=clin_feat_dim,
        num_t_classes=len(T_STAGES),
        num_n_classes=len(N_STAGES),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_pids, all_t_preds, all_n_preds = [], [], []

    with torch.no_grad():
        for x_img, x_clin, _, _ in loader:
            x_img, x_clin = x_img.to(device), x_clin.to(device)
            t_logits, n_logits = model(x_img, x_clin)
            t_preds = t_logits.argmax(dim=1).cpu().numpy()
            n_preds = n_logits.argmax(dim=1).cpu().numpy()
            all_t_preds.extend(t_encoder.inverse_transform(t_preds))
            all_n_preds.extend(n_encoder.inverse_transform(n_preds))

    all_pids = dataset.patient_ids
    results = pd.DataFrame({
        "PatientID": all_pids,
        "T_stage_pred": all_t_preds,
        "N_stage_pred": all_n_preds,
    })
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HECKTOR 2026 TN Staging Inference")
    parser.add_argument("--csv", required=True, help="Path to test CSV file")
    parser.add_argument("--images_dir", required=True, help="Path to test images directory")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--scaler_file", required=True, help="Path to saved StandardScaler (.pkl)")
    parser.add_argument("--ohe_file", required=True, help="Path to saved OneHotEncoder (.pkl)")
    parser.add_argument("--output_path", required=True, help="Path to save predictions CSV")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    run_inference(
        csv_path=args.csv,
        img_dir=args.images_dir,
        checkpoint=args.checkpoint,
        scaler_file=args.scaler_file,
        ohe_file=args.ohe_file,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
