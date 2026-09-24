from glob import glob
from os import scandir, listdir
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from monai.data import PersistentDataset
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureTyped,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    Lambdad,
    RandCropByPosNegLabeld,
    MapTransform,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandBiasFieldd,
    Rand3DElasticd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    Orientationd,
)
from utils.shared import get_dataset_filepaths


class ISLESDataset(PersistentDataset):
    # mask_add_bgc: whether to add a background channel to the target mask
    def __init__(
        self,
        split="train",
        range=None,
        mask_add_bgc=True,
        random_crop=False,
        domain_augment=False,
        random_seed=42,
        cache_dir="./monai_cache",
    ):
        self.mask_add_bgc = mask_add_bgc
        self.random_crop = random_crop

        self.metadata, self.features, self.labels = get_dataset_filepaths(
            f"./data/{split}", range
        )

        self.parsed_metadata = [self.parse_metadata(file) for file in self.metadata]

        data_dicts = [
            {"image": img, "mask": lbl, "metadata": meta}
            for img, lbl, meta in zip(self.features, self.labels, self.parsed_metadata)
        ]

        print(len(self.features))

        fixed_transforms = [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            Spacingd(
                keys=["image", "mask"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "mask"], axcodes="RAS"),
            ResizeWithPadOrCropd(
                keys=["image", "mask"],
                spatial_size=(256, 256, 256),
                mode="constant",
            ),
        ]

        domain_aug_transforms = [
            # RandBiasFieldd(
            #     keys=["image"],
            #     degree=3,
            #     coeff_range=(0.0, 0.1),
            #     prob=0.8,
            # ),
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.2,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.8),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.8),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.8),
            RandRotated(
                keys=["image", "label"],
                range_x=0.4,  # rotation range in radians
                range_y=0.4,
                range_z=0.4,
                mode=[
                    "trilinear",
                    "nearest",
                ],
                prob=0.8,
            ),
            # 2. Intensity: Apply ONLY to image
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.8),
            RandAdjustContrastd(
                keys=["image"],
                gamma=(0.5, 2.0),  # Contrast adjustment range
                prob=0.8,
            ),
            RandShiftIntensityd(keys=["image"], prob=0.8, offsets=0.1),
        ]

        super().__init__(
            data=data_dicts,
            transform=Compose(fixed_transforms),
            cache_dir=cache_dir,
        )

        dynamic_transforms = []

        if domain_augment:
            dynamic_transforms.extend(domain_aug_transforms)

        self.dynamic_transforms = Compose(dynamic_transforms)

    def parse_metadata(self, filepath):
        # 0: days_post_stroke missing? 0/1
        # 1: chronicity missing? 0/1
        # 2: days_post_stroke: float or nan
        # 3: chronicity: 0/1/2 or nan
        out = torch.empty((4), dtype=torch.float32)
        meta = pd.read_csv(filepath)
        if len(meta) > 0:
            dps = meta["DAYS_POST_STROKE"][0]
            chronicity = meta["CHRONICITY"][0]

            # print(type(chronicity))

            if np.isnan(dps):
                out[0] = 1.0
                out[2] = 0.0
            else:
                out[0] = 0.0
                out[2] = dps

            if np.isnan(chronicity):
                out[1] = 1.0
                out[3] = 0.0
            else:
                out[1] = 0.0
                out[3] = float(chronicity)

        else:
            out[0] = 1.0
            out[1] = 1.0
            out[2] = 0.0
            out[3] = 0.0

        return out

    def __getitem__(self, idx):
        data_dict = super().__getitem__(idx)
        out = self.dynamic_transforms(data_dict)

        if self.random_crop:
            return out[0]
        else:
            return out

    def __len__(self):
        return len(self.features)
