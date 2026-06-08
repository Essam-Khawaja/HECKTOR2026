from monai.data import DataLoader, CacheDataset
from .transforms import get_train_transforms, get_validation_transforms
from typing import Tuple, List, Dict
import json
import os


def load_splits(splits_file: str) -> List[Dict[str, List[str]]]:
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"Splits file not found: {splits_file}")
    with open(splits_file, 'r') as f:
        return json.load(f)


def get_dataloaders(config, fold: int = 0) -> Tuple[DataLoader, DataLoader]:
    splits = load_splits(config.splits_file)
    if fold >= len(splits):
        raise ValueError(f"Fold {fold} is out of range.")

    train_ids = splits[fold]['train']
    val_ids   = splits[fold]['val']
    print(f"Using fold {fold}: {len(train_ids)} training cases, {len(val_ids)} validation cases")

    def case_files(case_id: str) -> Dict[str, str]:
        case_dir = os.path.join(config.data_root, case_id)
        return {
            "ct": os.path.join(case_dir, f"{case_id}__CT.nii.gz"),
            "pet": os.path.join(case_dir, f"{case_id}__PT.nii.gz"),
            "label": os.path.join(case_dir, f"{case_id}.nii.gz"),
        }

    train_files = [case_files(case_id) for case_id in train_ids]
    val_files = [case_files(case_id) for case_id in val_ids]

    train_transforms = get_train_transforms(config)
    print(f"Creating CacheDataset for training with cache_rate={config.cache_rate}...")
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=config.cache_rate,
        num_workers=config.num_workers,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )

    val_transforms = get_validation_transforms()
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=config.cache_rate)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    return train_loader, val_loader
