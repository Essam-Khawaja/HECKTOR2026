"""
Simple U-Net for PET/CT segmentation.

Input:  2-channel image (PET channel + CT channel, each 1-channel, stacked)
Output: 1-channel segmentation mask (logits â€” apply sigmoid for probabilities)

Usage:
    model = UNet(in_channels=2, out_channels=1)
    pet = torch.randn(batch, 1, H, W)
    ct  = torch.randn(batch, 1, H, W)
    x = torch.cat([pet, ct], dim=1)   # -> (batch, 2, H, W)
    mask_logits = model(x)            # -> (batch, 1, H, W)
"""

import os
import glob

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling: maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Upscaling then double conv, with skip connection concat"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad if shapes mismatch due to odd input sizes
        diff_y = skip.size()[2] - x.size()[2]
        diff_x = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Simple U-Net.

    in_channels: number of input channels (2 for PET + CT, each 1-channel, stacked)
    out_channels: number of output channels (1 for binary mask)
    base_channels: number of channels after the first conv block (controls model size)
    """

    def __init__(self, in_channels=2, out_channels=1, base_channels=32, bilinear=True):
        super().__init__()

        c = base_channels

        # Encoder
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)

        # Decoder
        self.up1 = Up(c * 16 + c * 8, c * 8, bilinear)
        self.up2 = Up(c * 8 + c * 4, c * 4, bilinear)
        self.up3 = Up(c * 4 + c * 2, c * 2, bilinear)
        self.up4 = Up(c * 2 + c, c, bilinear)

        # Output head
        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)

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

        logits = self.outc(x)  # (B, out_channels, H, W)
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


class PETCTSegDataset(Dataset):
    """
    2D slice-wise PET/CT segmentation dataset backed by NIfTI files.

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

    Each item is a single axial slice:
        image: (2, H, W) float32  -> channel 0 = PET, channel 1 = CT
        mask:  (1, H, W) float32  -> binary mask (0/1)

    Slices that contain no foreground can optionally be skipped via
    `skip_empty_masks=True`, which is useful when lesions are sparse.
    """

    def __init__(
        self,
        root_dir,
        pet_filename="pet.nii.gz",
        ct_filename="ct.nii.gz",
        mask_filename="mask.nii.gz",
        skip_empty_masks=False,
        transform=None,
    ):
        self.transform = transform

        all_cases = sorted(
            d for d in glob.glob(os.path.join(root_dir, "*"))
            if os.path.isdir(d)
        )

        if not all_cases:
            raise ValueError(f"No case folders found under {root_dir}")

        self.pet_filename = pet_filename
        self.ct_filename = ct_filename
        self.mask_filename = mask_filename

        self._volume_cache = {}
        self.cases = []
        self.index = []

        skipped_cases = []

        for case_dir in all_cases:
            try:
                pet, ct, mask = self._load_case(case_dir)
            except Exception as e:
                skipped_cases.append((case_dir, str(e)))
                print(f"Skipping case {os.path.basename(case_dir)}: {e}")
                continue

            case_idx = len(self.cases)
            self.cases.append(case_dir)

            n_slices = pet.shape[0]

            for slice_idx in range(n_slices):
                if skip_empty_masks and mask[slice_idx].sum() == 0:
                    continue
                self.index.append((case_idx, slice_idx))

        print(f"Loaded valid cases: {len(self.cases)}")
        print(f"Skipped cases: {len(skipped_cases)}")

        if len(self.index) == 0:
            raise ValueError("No usable slices found. Check file paths, masks, or skip_empty_masks.")

    def _load_case(self, case_dir):
        if case_dir in self._volume_cache:
            return self._volume_cache[case_dir]

        case_id = os.path.basename(case_dir)

        # Use the preprocessed folder inside each patient/case folder
        data_dir = os.path.join(case_dir, "preprocessed")

        pet_path = os.path.join(data_dir, f"{case_id}__PT.nii.gz")
        ct_path = os.path.join(data_dir, f"{case_id}__CT.nii.gz")
        mask_path = os.path.join(data_dir, f"{case_id}.nii.gz")

        required_paths = {
            "PET": pet_path,
            "CT": ct_path,
            "MASK": mask_path,
        }

        for name, path in required_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} file: {path}")

        # pet = load_nifti_as_array(os.path.join(case_dir, self.pet_filename))
        # ct = load_nifti_as_array(os.path.join(case_dir, self.ct_filename))
        # mask = load_nifti_as_array(os.path.join(case_dir, self.mask_filename))

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

        self._volume_cache[case_dir] = (pet, ct, mask)
        return pet, ct, mask

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        case_idx, slice_idx = self.index[idx]
        case_dir = self.cases[case_idx]
        pet, ct, mask = self._load_case(case_dir)

        pet_slice = pet[slice_idx]    # (H, W)
        ct_slice = ct[slice_idx]      # (H, W)
        mask_slice = mask[slice_idx]  # (H, W)

        image = np.stack([pet_slice, ct_slice], axis=0)  # (2, H, W)
        mask_out = mask_slice[None, ...]                 # (1, H, W)

        image = torch.from_numpy(image).float()
        mask_out = torch.from_numpy(mask_out).float()

        if self.transform is not None:
            image, mask_out = self.transform(image, mask_out)

        return image, mask_out


def get_dataloaders(
    train_dir,
    val_dir=None,
    batch_size=8,
    num_workers=4,
    skip_empty_masks_train=True,
):
    """Convenience helper to build train/val DataLoaders from NIfTI case folders."""
    train_ds = PETCTSegDataset(train_dir, skip_empty_masks=skip_empty_masks_train)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = None
    if val_dir is not None:
        val_ds = PETCTSegDataset(val_dir, skip_empty_masks=False)
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Model sanity check with random tensors
    batch_size, height, width = 2, 128, 128

    model = UNet(in_channels=2, out_channels=1, base_channels=32)

    pet = torch.randn(batch_size, 1, height, width)
    ct = torch.randn(batch_size, 1, height, width)
    x = torch.cat([pet, ct], dim=1)  # (B, 2, H, W)

    mask_logits = model(x)
    mask_probs = torch.sigmoid(mask_logits)

    print("Model sanity check")
    print("  Input shape: ", x.shape)
    print("  Output shape:", mask_logits.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # 2. Example of wiring up the NIfTI dataloader (uncomment and point at
    #    your actual data directory laid out as described in PETCTSegDataset):
    data_root = "/home/syedessamuddin.khawa/HECKTOR 2026 Training Data"

    full_dataset = PETCTSegDataset(
        root_dir=data_root,
        skip_empty_masks=True
    )

    print("Total usable slices:", len(full_dataset))

    val_fraction = 0.2

    val_size = int(len(full_dataset) * val_fraction)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print("Train slices:", len(train_dataset))
    print("Val slices:  ", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    images, masks = next(iter(train_loader))

    print("Batch images:", images.shape)  # should be (B, 2, H, W)
    print("Batch masks: ", masks.shape)   # should be (B, 1, H, W)

    logits = model(images)

    print("Model output:", logits.shape)

    loss = F.binary_cross_entropy_with_logits(logits, masks)

    print("Example loss:", loss.item())

    # train_loader, val_loader = get_dataloaders(
    #     train_dir="/path/to/data/train",
    #     val_dir="/path/to/data/val",
    #     batch_size=8,
    # )
    # images, masks = next(iter(train_loader))
    # print("Batch images:", images.shape)  # (B, 2, H, W)
    # print("Batch masks: ", masks.shape)   # (B, 1, H, W)
    # logits = model(images)
    # loss = F.binary_cross_entropy_with_logits(logits, masks)
    # print("Example loss:", loss.item())