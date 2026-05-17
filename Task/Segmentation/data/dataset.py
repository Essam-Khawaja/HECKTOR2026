"""HECKTOR dataset implementation."""

import os
import glob
from typing import Dict, List, Optional, Callable
import torch
from torch.utils.data import Dataset
from monai.transforms import Compose


class HecktorDataset(Dataset):
    """HECKTOR dataset for 3D segmentation."""

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
        split: str = "train",
        case_ids: Optional[List[str]] = None
    ):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.split = split
        self.case_ids = case_ids if case_ids is not None else self._get_case_ids()

    def _get_case_ids(self) -> List[str]:
        ct_files = glob.glob(os.path.join(self.images_dir, "*__CT.nii.gz"))
        case_ids = [os.path.basename(f).replace("__CT.nii.gz", "") for f in ct_files]
        return sorted(case_ids)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_id = self.case_ids[idx]
        data = {
            "ct":    os.path.join(self.images_dir, f"{case_id}__CT.nii.gz"),
            "pet":   os.path.join(self.images_dir, f"{case_id}__PT.nii.gz"),
            "label": os.path.join(self.labels_dir, f"{case_id}.nii.gz"),
            "case_id": case_id,
        }
        if self.transform:
            data = self.transform(data)
        return data
