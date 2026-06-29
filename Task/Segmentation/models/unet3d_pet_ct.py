"""
Simple 3D U-Net for PET/CT segmentation.

Input:  2-channel volume (PET channel + CT channel, each 1-channel, stacked)
Output: 1-channel segmentation mask (logits â€” apply sigmoid for probabilities)

Usage:
    model = UNet3D(in_channels=2, out_channels=1)
    pet = torch.randn(batch, 1, D, H, W)
    ct  = torch.randn(batch, 1, D, H, W)
    x = torch.cat([pet, ct], dim=1)   # -> (batch, 2, D, H, W)
    mask_logits = model(x)            # -> (batch, 1, D, H, W)

This file mirrors unet_pet_ct.py but operates on full 3D volumes instead of
2D slices. Because full-resolution 3D volumes rarely fit in GPU memory, the
dataset extracts fixed-size random patches during training and fixed-size
center patches during validation. Swap in a sliding-window inference scheme
for full-volume prediction at test time.
"""

import os
import glob

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DoubleConv3D(nn.Module):
    """(Conv3d -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """Downscaling: maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    """Upscaling then double conv, with skip connection concat"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if shapes mismatch due to odd input sizes
        diff_d = skip.size()[2] - x.size()[2]
        diff_y = skip.size()[3] - x.size()[3]
        diff_x = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2,
                       diff_d // 2, diff_d - diff_d // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Simple 3D U-Net.

    in_channels: number of input channels (2 for PET + CT, each 1-channel, stacked)
    out_channels: number of output channels (1 for binary mask)
    base_channels: number of channels after the first conv block (controls model size)

    Note: 3D convs are memory-hungry. base_channels=16-32 with patch sizes
    around 96-128 per side is a reasonable starting point on a single GPU.
    """

    def __init__(self, in_channels=2, out_channels=1, base_channels=16, trilinear=True):
        super().__init__()

        c = base_channels

        # Encoder
        self.inc = DoubleConv3D(in_channels, c)
        self.down1 = Down3D(c, c * 2)
        self.down2 = Down3D(c * 2, c * 4)
        self.down3 = Down3D(c * 4, c * 8)
        self.down4 = Down3D(c * 8, c * 16)

        # Decoder
        self.up1 = Up3D(c * 16 + c * 8, c * 8, trilinear)
        self.up2 = Up3D(c * 8 + c * 4, c * 4, trilinear)
        self.up3 = Up3D(c * 4 + c * 2, c * 2, trilinear)
        self.up4 = Up3D(c * 2 + c, c, trilinear)

        # Output head
        self.outc = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # full res
        x2 = self.down1(x1)    # /2
        x3 = self.down2(x2)    # /4
        x4 = self.down3(x3)    # /8
        x5 = self.down4(x4)    # /16

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)  # (B, out_channels, D, H, W)
        return logits


# ---------------------------------------------------------------------------
# Data loading (NIfTI, via SimpleITK)
# ---------------------------------------------------------------------------

def load_nifti_as_array(path):
    """Load a NIfTI file and return it as a numpy array with shape (Z, Y, X)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # SimpleITK returns (Z, Y, X)
    return arr.astype(np.float32)


def normalize_ct(arr, min_hu=-1000.0, max_hu=1000.0):
    """Clip CT to a Hounsfield-unit window and scale to [0, 1]."""
    arr = np.clip(arr, min_hu, max_hu)
    arr = (arr - min_hu) / (max_hu - min_hu)
    return arr


def normalize_pet(arr, percentile=99.5):
    """Scale PET (e.g. SUV) to [0, 1] using a high percentile to avoid outlier hotspots."""
    upper = np.percentile(arr, percentile)
    upper = max(upper, 1e-6)
    arr = np.clip(arr, 0, upper) / upper
    return arr


def pad_to_min_size(arr, min_size):
    """Zero-pad a (Z, Y, X) array so each dim is at least min_size. Returns padded array."""
    pad = []
    for dim, m in zip(arr.shape, min_size):
        total = max(0, m - dim)
        pad.append((total // 2, total - total // 2))
    return np.pad(arr, pad, mode="constant", constant_values=0)


def random_crop_3d(volumes, patch_size, rng):
    """
    Randomly crop the same spatial window out of a list of (Z, Y, X) arrays
    that all share the same shape (e.g. [pet, ct, mask]).
    """
    z, y, x = volumes[0].shape
    pz, py, px = patch_size

    z0 = rng.integers(0, max(z - pz, 0) + 1)
    y0 = rng.integers(0, max(y - py, 0) + 1)
    x0 = rng.integers(0, max(x - px, 0) + 1)

    return [v[z0:z0 + pz, y0:y0 + py, x0:x0 + px] for v in volumes]


def center_crop_3d(volumes, patch_size):
    """Center-crop the same spatial window out of a list of (Z, Y, X) arrays."""
    z, y, x = volumes[0].shape
    pz, py, px = patch_size

    z0 = max((z - pz) // 2, 0)
    y0 = max((y - py) // 2, 0)
    x0 = max((x - px) // 2, 0)

    return [v[z0:z0 + pz, y0:y0 + py, x0:x0 + px] for v in volumes]


class PETCTSegDataset3D(Dataset):
    """
    3D patch-based PET/CT segmentation dataset backed by NIfTI files.

    Expected directory layout, one sub-folder per case:

        root_dir/
            case_001/
                pet.nii.gz
                ct.nii.gz
                mask.nii.gz
            case_002/
                pet.nii.gz
                ct.nii.gz
                mask.nii.gz
            ...

    PET, CT, and mask volumes for a case must share the same geometry
    (size, spacing, origin) â€” resample beforehand if they don't.

    Each item is a fixed-size 3D patch:
        image: (2, pz, py, px) float32 -> channel 0 = PET, channel 1 = CT
        mask:  (1, pz, py, px) float32 -> binary mask (0/1)

    Training patches are randomly located (re-sampled every epoch via
    `__getitem__`); validation patches are taken from the volume center.
    For full-volume inference, write a separate sliding-window function
    rather than reusing this dataset.
    """

    def __init__(
        self,
        root_dir,
        patch_size=(64, 128, 128),
        pet_filename="pet.nii.gz",
        ct_filename="ct.nii.gz",
        mask_filename="mask.nii.gz",
        mode="train",  # "train" -> random crop, "val"/"test" -> center crop
        seed=0,
    ):
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)

        self.cases = sorted(
            d for d in glob.glob(os.path.join(root_dir, "*"))
            if os.path.isdir(d)
        )

        if not self.cases:
            raise ValueError(f"No case folders found under {root_dir}")

        self.pet_filename = pet_filename
        self.ct_filename = ct_filename
        self.mask_filename = mask_filename

        # Keep only cases that actually have the expected HECKTOR preprocessed files.
        valid_cases = []
        skipped_cases = []

        for case_dir in self.cases:
            case_id = os.path.basename(case_dir)
            data_dir = os.path.join(case_dir, "preprocessed")

            pet_path = os.path.join(data_dir, f"{case_id}__PT.nii.gz")
            ct_path = os.path.join(data_dir, f"{case_id}__CT.nii.gz")
            mask_path = os.path.join(data_dir, f"{case_id}.nii.gz")

            if os.path.exists(pet_path) and os.path.exists(ct_path) and os.path.exists(mask_path):
                valid_cases.append(case_dir)
            else:
                skipped_cases.append(case_id)

        self.cases = valid_cases

        print(f"Loaded valid 3D cases: {len(self.cases)}")
        print(f"Skipped 3D cases: {len(skipped_cases)}")

        if not self.cases:
            raise ValueError("No valid cases found. Check the file paths/names.")

        # Cache loaded (and normalized/padded) volumes per case.
        self._volume_cache = {}


    def _load_case(self, case_dir):
        if case_dir in self._volume_cache:
            return self._volume_cache[case_dir]

        case_id = os.path.basename(case_dir)
        data_dir = os.path.join(case_dir, "preprocessed")

        pet_path = os.path.join(data_dir, f"{case_id}__PT.nii.gz")
        ct_path = os.path.join(data_dir, f"{case_id}__CT.nii.gz")
        mask_path = os.path.join(data_dir, f"{case_id}.nii.gz")

        pet = load_nifti_as_array(pet_path)
        ct = load_nifti_as_array(ct_path)
        mask = load_nifti_as_array(mask_path)

        if not (pet.shape == ct.shape == mask.shape):
            raise ValueError(
                f"Shape mismatch in {case_dir}: "
                f"pet={pet.shape}, ct={ct.shape}, mask={mask.shape}"
            )

        pet = normalize_pet(pet)
        ct = normalize_ct(ct)
        mask = (mask > 0).astype(np.float32)

        # Ensure every volume is at least as large as the patch size so
        # cropping never goes out of bounds.
        pet = pad_to_min_size(pet, self.patch_size)
        ct = pad_to_min_size(ct, self.patch_size)
        mask = pad_to_min_size(mask, self.patch_size)

        self._volume_cache[case_dir] = (pet, ct, mask)
        return pet, ct, mask

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_dir = self.cases[idx]
        pet, ct, mask = self._load_case(case_dir)

        if self.mode == "train":
            pet_p, ct_p, mask_p = random_crop_3d([pet, ct, mask], self.patch_size, self.rng)
        else:
            pet_p, ct_p, mask_p = center_crop_3d([pet, ct, mask], self.patch_size)

        image = np.stack([pet_p, ct_p], axis=0)  # (2, pz, py, px)
        mask_out = mask_p[None, ...]             # (1, pz, py, px)

        image = torch.from_numpy(image.copy()).float()
        mask_out = torch.from_numpy(mask_out.copy()).float()

        return image, mask_out


def get_dataloaders_3d(
    train_dir,
    val_dir=None,
    patch_size=(64, 128, 128),
    batch_size=2,
    num_workers=4,
):
    """Convenience helper to build train/val 3D DataLoaders from NIfTI case folders."""
    train_ds = PETCTSegDataset3D(train_dir, patch_size=patch_size, mode="train")
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if val_dir is not None:
        val_ds = PETCTSegDataset3D(val_dir, patch_size=patch_size, mode="val")
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    """
    Computes Dice score for binary segmentation.

    logits:  raw model output, shape (B, 1, D, H, W)
    targets: ground truth mask, shape (B, 1, D, H, W), values 0 or 1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3, 4))
    pred_sum = preds.sum(dim=(1, 2, 3, 4))
    target_sum = targets.sum(dim=(1, 2, 3, 4))

    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)

    return dice.mean()


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Model sanity check with random tensors
    batch_size, depth, height, width = 1, 32, 64, 64

    model = UNet3D(in_channels=2, out_channels=1, base_channels=16)

    pet = torch.randn(batch_size, 1, depth, height, width)
    ct = torch.randn(batch_size, 1, depth, height, width)
    x = torch.cat([pet, ct], dim=1)  # (B, 2, D, H, W)

    mask_logits = model(x)

    print("Model sanity check")
    print("  Input shape: ", x.shape)
    print("  Output shape:", mask_logits.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # 2. Wire up the real HECKTOR dataset
    data_root = "/home/syedessamuddin.khawa/HECKTOR 2026 Training Data"

    patch_size = (32, 96, 128)
    batch_size = 1

    train_dataset = PETCTSegDataset3D(
        root_dir=data_root,
        patch_size=patch_size,
        mode="train",
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    images, masks = next(iter(train_loader))

    print("Batch images:", images.shape)  # (B, 2, D, H, W)
    print("Batch masks: ", masks.shape)   # (B, 1, D, H, W)

    logits = model(images)

    print("Model output:", logits.shape)

    loss = F.binary_cross_entropy_with_logits(logits, masks)
    dice = dice_score_from_logits(logits, masks)

    print("Example loss:", loss.item())
    print("Example Dice:", dice.item())

    # 2. Example of wiring up the NIfTI dataloader (uncomment and point at
    #    your actual data directory laid out as described in PETCTSegDataset3D):
    #
    # train_loader, val_loader = get_dataloaders_3d(
    #     train_dir="/path/to/data/train",
    #     val_dir="/path/to/data/val",
    #     patch_size=(64, 128, 128),
    #     batch_size=2,
    # )
    # images, masks = next(iter(train_loader))
    # print("Batch images:", images.shape)  # (B, 2, pz, py, px)
    # print("Batch masks: ", masks.shape)   # (B, 1, pz, py, px)
    # logits = model(images)
    # loss = F.binary_cross_entropy_with_logits(logits, masks)
    # print("Example loss:", loss.item())