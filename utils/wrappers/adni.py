import os
from os import scandir, DirEntry
from typing import List
import pandas as pd
import numpy as np


def get_image_dirs(root: str) -> List[DirEntry]:
    dirs = []

    return dirs


def get_image_paths(root: str, range: tuple = None) -> list:

    ret = []

    for subject in scandir(root):
        for image in scandir(subject.path):
            files = list(os.scandir(image.path))
            filenames = [file.name for file in files]

            if "temp.nii.gz" in filenames:
                continue

            if "img.nii.gz" in filenames and "img_seg.nii.gz" in filenames:
                ret.append(
                    {
                        "mri_path": os.path.join(image.path, "img.nii.gz"),
                        "seg_path": os.path.join(image.path, "img_seg.nii.gz"),
                        "image_id": int(image.name),
                        "subject_id": subject.name,
                    }
                )

    if range is not None:
        ret = ret[range[0] : range[1]]

    return ret


# Loads and sanitizes the demographics file


def load_demographics(path: str, range: tuple = None) -> pd.DataFrame:
    """
    Loads and sanitizes the demographics file.
    Converts into floats usable for training.
    """
    demogs = pd.read_csv(
        path,
        index_col="PTID",
        usecols=[
            "PTID",
            "PTGENDER",
            "PTDOBYY",
            "PTEDUC",
            "PTRACCAT",
            "PTETHCAT",
            "PTMARRY",
            "PTHAND",
        ],
        dtype={
            "PTGENDER": np.float32,
            "PTDOBYY": np.str_,  # parse later into a float
            "PTEDUC": np.float32,
            "PTRACCAT": np.float32,  # 7 = unknown/missing
            "PTETHCAT": np.float32,  # 3 = unknown/missing
            "PTMARRY": np.float32,  # 5 = unknown/missing
            "PTHAND": np.float32,
        },
        na_values=[-4, -1, -4.0, -1.0, "-4", "-1"],
    )

    # Converts birth year into float
    demogs["PTDOBYY"] = demogs["PTDOBYY"].str[:4].astype(np.float32)

    # Sets unknown values to pd.NA
    demogs["PTRACCAT"] = demogs["PTRACCAT"].replace(7, pd.NA)
    demogs["PTETHCAT"] = demogs["PTETHCAT"].replace(3, pd.NA)
    demogs["PTMARRY"] = demogs["PTMARRY"].replace(5, pd.NA)

    demogs.rename(
        {
            "PTID": "sid",      # subject id
            "PTGENDER": "gdr",  # gender
            "PTDOBYY": "dob",   # year of birth
            "PTEDUC": "edc",    # education years
            "PTRACCAT": "rac",  # racial category
            "PTETHCAT": "eth",  # ethnic category
            "PTMARRY": "mts",   # marital status
            "PTHAND": "hnd",    # handedness
        }
    ) 
    
    # Introduces categorical variables to denote
    # missing values to avoid confusing the model:
    
    demogs["gdr_mis"] = 0.
    demogs["dob_mis"] = 0.
    demogs["edc_mis"] = 0.
    demogs["rac_mis"] = 0.
    demogs["eth_mis"] = 0.
    demogs["mts_mis"] = 0.
    demogs["hnd_mis"] = 0.
    
    demogs[demogs["gdr"] == pd.NA]["gdr_mis"] = 1.
    demogs[demogs["dob"] == pd.NA]["dob_mis"] = 1.
    demogs[demogs["edc"] == pd.NA]["edc_mis"] = 1.
    demogs[demogs["rac"] == pd.NA]["rac_mis"] = 1.
    demogs[demogs["eth"] == pd.NA]["eth_mis"] = 1.
    demogs[demogs["mts"] == pd.NA]["mts_mis"] = 1.
    demogs[demogs["hnd"] == pd.NA]["hnd_mis"] = 1.
    
    return demogs


def get_img_lbl_paths(
    root: str, subject_id: str, image_id: int, check_exists: bool = True
) -> list:

    img_path = os.path.join(root, subject_id, image_id, "img.nii.gz")
    seg_path = os.path.join(root, subject_id, image_id, "img_seg.nii.gz")

    if check_exists:
        assert os.path.exists(img_path), "Source image not found."
        assert os.path.exists(seg_path), "Source segmentation path not found."

    return img_path, seg_path
