"""Base configuration class for all models."""

import os
from dataclasses import dataclass
from typing import Tuple


def _repo_root() -> str:
    """Return the repository root from this config file."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _segmentation_root() -> str:
    """Return the segmentation task root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def default_data_root() -> str:
    """Resolve the default HECKTOR training data path.

    Priority:
    1. HECKTOR_DATA_ROOT environment variable.
    2. The ARC home-level data folder: ../../"HECKTOR 2026 Training Data".
    3. A sibling folder named "HECKTOR 2026 Training Data".
    4. The challenge-style folder inside the repo, if present.
    """
    env_data_root = os.environ.get("HECKTOR_DATA_ROOT")
    if env_data_root:
        return os.path.abspath(os.path.expanduser(env_data_root))

    repo_root = _repo_root()
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(repo_root)), "HECKTOR 2026 Training Data"),
        os.path.join(os.path.dirname(repo_root), "HECKTOR 2026 Training Data"),
        os.path.join(repo_root, "HECKTOR 2026 Training Data"),
        os.path.join(repo_root, "hecktor2026_training"),
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    return candidates[0]


@dataclass
class BaseConfig:
    """Base configuration class with common parameters."""
    
    # Data paths - per-patient folders: {data_root}/{PatientID}/{PatientID}__CT.nii.gz etc.
    data_root: str = ""
    splits_file: str = "config/splits_final.json"
    
    # Data properties
    input_channels: int = 2  # CT + PET
    num_classes: int = 3     # background + primary tumor + metastatic tumor
    spatial_size: Tuple[int, int, int] = (128, 128, 128)
    
    # Training parameters
    batch_size: int = 2
    learning_rate: float = 1e-2
    weight_decay: float = 3e-5
    num_epochs: int = 350
    
    # Scheduler parameters
    # PolyLR scheduler parameters
    poly_lr_power: float = 0.9
    poly_lr_min_lr: float = 1e-6
    
    # Data augmentation
    use_augmentation: bool = True
    aug_probability: float = 0.5
    rotation_range: float = 15.0
    scaling_range: float = 0.1
    translation_range: float = 10.0
    crop_num_samples: int = 3
    
    # System parameters
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data caching parameters
    cache_rate: float = 0.25  # Cache 25% of training data for faster loading
    
    # Checkpointing and logging
    save_checkpoint_every: int = 1 # Save checkpoint every n epochs
    use_tensorboard: bool = True
    
    # Output directories
    experiment_name: str = "baseline"
    output_dir: str = "experiments"
    fold: int = 0
    
    def __post_init__(self):
        """Setup output directories with fold-specific structure."""
        if not self.data_root:
            self.data_root = default_data_root()
        else:
            self.data_root = os.path.abspath(os.path.expanduser(self.data_root))

        if not os.path.isabs(self.splits_file):
            self.splits_file = os.path.join(_segmentation_root(), self.splits_file)

        self.setup_output_dirs()

    def setup_output_dirs(self):
        """Setup output directories with fold-specific structure."""
        # Create fold-specific directory structure
        self.experiment_dir = os.path.join(self.output_dir, self.experiment_name)
        self.fold_dir = os.path.join(self.experiment_dir, f"fold_{self.fold}")
        self.checkpoint_dir = os.path.join(self.fold_dir, "checkpoints")
        self.log_dir = os.path.join(self.fold_dir, "logs")
        
        # Create directories
        for dir_path in [self.experiment_dir, self.fold_dir, self.checkpoint_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
