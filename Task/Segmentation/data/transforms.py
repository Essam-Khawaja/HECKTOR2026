"""Data transforms for HECKTOR dataset."""

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
    RandCropByLabelClassesd,
    ConcatItemsd,
    SelectItemsd,
    CropForegroundd,
)


def get_train_transforms(config):
    keys = ["ct", "pet", "label"]

    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        # CT: window to soft-tissue range and scale to [0, 1]
        ScaleIntensityRanged(
            keys=["ct"], a_min=-175, a_max=275,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # PET: z-score normalise SUV values
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="ct"),
        RandCropByLabelClassesd(
            keys=keys,
            label_key="label",
            spatial_size=config.spatial_size,
            ratios=[0.1, 0.45, 0.45],
            num_classes=3,
            num_samples=3,
            allow_missing_keys=True,
            warn=False,
        ),
    ]

    if config.use_augmentation:
        transforms += [
            RandFlipd(keys=keys, spatial_axis=[0, 1, 2], prob=config.aug_probability),
            RandScaleIntensityd(keys=["ct"], factors=0.1, prob=config.aug_probability),
            RandShiftIntensityd(keys=["ct"], offsets=0.1, prob=config.aug_probability),
            RandGaussianNoised(keys=["ct"], std=0.01, prob=config.aug_probability),
            RandGaussianSmoothd(
                keys=["ct"],
                sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15),
                prob=config.aug_probability,
            ),
        ]

    transforms += [
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ]

    return Compose(transforms)


def get_validation_transforms():
    keys = ["ct", "pet", "label"]

    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["ct"], a_min=-175, a_max=275,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=keys, source_key="ct"),
        ConcatItemsd(keys=["ct", "pet"], name="image", dim=0),
        SelectItemsd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
