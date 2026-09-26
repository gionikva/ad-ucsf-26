from glob import glob
from os import scandir, listdir
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import (
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotated,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandShiftIntensityd,
    RandBiasFieldd,
    Rand3DElasticd,
)
from utils.wrappers.adni import get_image_paths, load_demographics
from sklearn.model_selection import train_test_split


class ADNISegDataset(Dataset):
    """Dataset for the ADNI segmentation pretraining task.

        Args:
            images_path (str, optional): 
                Root directory of the preprocessed ADNI mri/segmentation dataset.
                Defaults to ``"./data/images"``.
            split (Literal["train", "test", None], optional): 
                The data split (test/train). If None, will not split. Defaults to ``None``.
            test_ratio (float, optional): 
                The ratio of test to total images. Must be in ``(0, 1)``.
            range (tuple, optional):
                The index range of images to consider. 
                Note: if ``split`` is specified ``len(dataset) < (range[1] - range[0])``
            domain_augment (bool, optional):
                Whether to apply domain augmentations.
                Defaults to ``False``.
            random_seed (int, optional):
                Random seed for data shuffling.
                Defaults to ``42``.
    """
    # mask_add_bgc: whether to add a background channel to the target mask
    def __init__(
        self,
        images_path="./data/images",
        split=None, 
        test_ratio=0.1,
        range=None,
        domain_augment=False,
        random_seed=42,
    ):
        super().__init__()
        
        self.image_paths = get_image_paths(images_path, range)        

        # Data splitting 
        if split is not None:
            train, test = train_test_split(
                self.image_paths, test_size=test_ratio, random_state=random_seed
            )
        
            if split == "train":
                self.image_paths = train
            elif split == "test":
                self.image_paths = test

        image_key = "mri"
        label_key = "seg"

        transform_list = [
            LoadImaged(keys=[image_key, label_key]),
            EnsureChannelFirstd(keys=[image_key, label_key]),
        ]

        domain_aug_transforms = [
            RandBiasFieldd(
                keys=[image_key],
                degree=3,
                coeff_range=(0.0, 0.1),
                prob=0.8,
            ),
            Rand3DElasticd(
                keys=[image_key, label_key],
                prob=0.2,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandFlipd(keys=[image_key, label_key], spatial_axis=0, prob=0.8),
            RandFlipd(keys=[image_key, label_key], spatial_axis=1, prob=0.8),
            RandFlipd(keys=[image_key, label_key], spatial_axis=2, prob=0.8),
            RandRotated(
                keys=[image_key, label_key],
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
            RandGaussianNoised(keys=[image_key], mean=0.0, std=0.1, prob=0.8),
            RandAdjustContrastd(
                keys=[image_key],
                gamma=(0.5, 2.0),  # Contrast adjustment range
                prob=0.8,
            ),
            RandShiftIntensityd(keys=[image_key], prob=0.8, offsets=0.1),
        ]

        if domain_augment:
            transform_list.extend(domain_aug_transforms)

        self.transforms = Compose(transform_list)

    def __getitem__(self, idx):
        entry = self.image_paths[idx]
        
        dict_ = {
            "mri": entry["mri_path"],
            "seg": entry["seg_path"]
        }
        
        out = self.transforms(dict_)

        return out

    def __len__(self):
        return len(self.image_paths)
