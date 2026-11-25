from glob import glob
from os import scandir, listdir
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
    AsDiscreted,
)
from utils.wrappers.adni import get_image_paths, load_demographics
from sklearn.model_selection import train_test_split, GroupShuffleSplit


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
            AsDiscreted(
                keys=[label_key], to_onehot=4
            ),  # Conversion of fast output to 4-channel map
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

        dict_ = {"mri": entry["mri_path"], "seg": entry["seg_path"]}

        out = self.transforms(dict_)

        return out

    def __len__(self):
        return len(self.image_paths)

COG_COLUMNS = [f"q{i}" for i in range(1, 12)]  # matcher output
# ADNI codes to one-hot encode. Every listed code always gets a column, so
# the demographics vector has a fixed size (2 numeric + 14 one-hot = 16).
# Unlisted or missing codes give an all-zero block.
CATEGORIES = {
    "PTGENDER": [1, 2],  # Male, Female
    "PTETHCAT": [1, 2],  # Hispanic/Latino, Not Hispanic/Latino
    "PTRACCAT": [1, 2, 3, 4, 5, 6],  # Am. Indian, Asian, Hawaiian/PI, Black, White, >1 race
    "PTMARRY": [1, 2, 3, 4],  # Married, Widowed, Divorced, Never married
}


def load_tables(matches_path, demographics_path):
    """One row per image: subject, cognitive scores, and demographic features."""
    matches = pd.read_csv(matches_path, dtype={"image_id": str})
    matches["image_id"] = matches["image_id"].str.lstrip("Ii")
    matches["image_date"] = pd.to_datetime(matches["image_date"])

    codes = ["PTDOBYY", "PTEDUCAT", *CATEGORIES]
    demog = pd.read_csv(
        demographics_path,
        usecols=["PTID", "VISDATE", "PTDOB", *codes],
        dtype=str,
    )
    # Multi-select race answers (e.g. "4|5") count as more than one race.
    demog.loc[demog["PTRACCAT"].str.contains("|", regex=False, na=False), "PTRACCAT"] = "6"
    demog[codes] = demog[codes].apply(pd.to_numeric, errors="coerce")
    demog[codes] = demog[codes].mask(demog[codes] < 0)  # ADNI missing codes (-1, -4)
    demog["PTDOB"] = pd.to_datetime(demog["PTDOB"], format="mixed", errors="coerce")
    demog["VISDATE"] = pd.to_datetime(demog["VISDATE"], format="mixed", errors="coerce")
    # First non-missing value per field, earliest visit first.
    demog = demog.sort_values("VISDATE").groupby("PTID").first()

    df = matches.join(demog, on="subject_id", how="inner")

    # PTDOB holds month and year (parsed as the 1st); use mid-month. Fall back
    # to mid-year when only PTDOBYY is known.
    birth = (df["PTDOB"] + pd.Timedelta(days=14)).fillna(
        pd.to_datetime(df["PTDOBYY"].astype("Int64").astype(str) + "-07-01", errors="coerce")
    )
    numeric = pd.DataFrame({
        "AGE": (df["image_date"] - birth).dt.days / 365.25 / 100,
        "PTEDUCAT": df["PTEDUCAT"] / 20,
    })
    numeric = numeric.fillna(numeric.mean())
    onehot = pd.get_dummies(
        df[list(CATEGORIES)].astype("Int64").apply(
            lambda col: pd.Categorical(col, categories=CATEGORIES[col.name])
        )
    )
    dem = pd.concat([numeric, onehot], axis=1).astype("float32")

    return df.set_index("image_id"), dem.set_axis(df["image_id"])


class ADNICogDataset(Dataset):
    """Dataset for the ADNI cogscore prediction task.

    Items hold the MRI (``"mri"``), the 11 ADAS-Cog11 subscores from the
    matches table (``"cog"``), a 16-value demographics vector (``"dem"``,
    feature names in ``dem_names``), ``"image_id"`` and ``"subject_id"``.
    Images missing from the matches or demographics table are dropped.

    Args:
        images_path (str, optional):
            Root directory of the preprocessed ADNI mri/segmentation dataset.
            Defaults to ``"./data/images"``.
        matches_path (str, optional):
            CSV of MRI–ADAS-Cog11 pairs written by ``MRI_ADAS-Cog11_Matching.py``.
            Defaults to ``"./MRI_ADAS11_pairs.csv"``.
        demographics_path (str, optional):
            ADNI demographics table (PTDEMOG export) with ``PTID``.
            Defaults to ``"./data/tables/PTDEMOG.csv"``.
        split (Literal["train", "test", None], optional):
            The data split (test/train). If None, will not split. Defaults to ``None``.
            Splitting is by participant, so no participant is in both splits.
        test_ratio (float, optional):
            The ratio of test to total participants. Must be in ``(0, 1)``.
            The image ratio is approximate. Defaults to ``0.1``.
        range (tuple, optional):
            The index range of images to consider.
            Note: if ``split`` is specified ``len(dataset) < (range[1] - range[0])``
        cogscore_range (tuple, optional):
            Slice of the 11 subscores to return. Defaults to ``(0, 11)``.
        domain_augment (bool, optional):
            Whether to apply domain augmentations.
            Defaults to ``False``.
        random_seed (int, optional):
            Random seed for data shuffling.
            Defaults to ``42``.
    """

    def __init__(
        self,
        images_path="./data/images/train",
        matches_path="./data/tables/matches.csv",
        demographics_path="./data/tables/DEMOG.csv",
        split=None,
        test_ratio=0.1,
        range=None,
        cogscore_range=(0, 11),
        domain_augment=False,
        random_seed=42,
    ):
        super().__init__()
        self.cogscore_range = cogscore_range

        table, dem = load_tables(matches_path, demographics_path)
        self.dem_names = list(dem.columns)

        # get_image_paths gives integer image IDs; the table index is strings.
        entries = [
            {**e, "image_id": str(e["image_id"])}
            for e in get_image_paths(images_path, range)
            if str(e["image_id"]) in table.index
        ]

        if split is not None:
            groups = [e["subject_id"] for e in entries]
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
            train, test = next(splitter.split(entries, groups=groups))
            entries = [entries[i] for i in (train if split == "train" else test)]

        ids = [e["image_id"] for e in entries]
        self.image_paths = entries
        self.subject_ids = table.loc[ids, "subject_id"].tolist()
        self.cog = torch.tensor(table.loc[ids, COG_COLUMNS].to_numpy("float32"))
        self.dem = torch.tensor(dem.loc[ids].to_numpy())

        image_key = "mri"

        transform_list = [
            LoadImaged(keys=[image_key]),
            EnsureChannelFirstd(keys=[image_key]),
        ]

        domain_aug_transforms = [
            RandBiasFieldd(
                keys=[image_key],
                degree=3,
                coeff_range=(0.0, 0.1),
                prob=0.8,
            ),
            Rand3DElasticd(
                keys=[image_key],
                prob=0.2,
                sigma_range=(5, 8),
                magnitude_range=(100, 200),
                mode=("bilinear"),
                padding_mode="zeros",
            ),
            RandFlipd(keys=[image_key], spatial_axis=0, prob=0.8),
            RandFlipd(keys=[image_key], spatial_axis=1, prob=0.8),
            RandFlipd(keys=[image_key], spatial_axis=2, prob=0.8),
            RandRotated(
                keys=[image_key],
                range_x=0.4,  # rotation range in radians
                range_y=0.4,
                range_z=0.4,
                mode=("trilinear"),
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
        lo, hi = self.cogscore_range
        return self.transforms({
            "mri": self.image_paths[idx]["mri_path"],
            "cog": self.cog[idx, lo:hi],
            "dem": self.dem[idx],
            "image_id": self.image_paths[idx]["image_id"],
            "subject_id": self.subject_ids[idx],
        })

    def __len__(self):
        return len(self.image_paths)