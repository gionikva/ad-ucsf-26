from torch._tensor import Tensor
from ants.core.ants_image import ANTsImage
import os
from os import listdir
from tabnanny import verbose
from typing import Any
from pandas.core.frame import DataFrame
import subprocess
from argparse import ArgumentParser
from monai.data.meta_tensor import MetaTensor
import torch
import ants
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import monai
from monai.transforms import (
    Compose,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    CropForegroundd,
    LoadImaged,
    EnsureTyped,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    CropForeground,
    Spacing,
    Orientation,
)
from monai.data import NibabelWriter
import nibabel as nib
import pandas as pd
import random

from utils.preprocessing import dcm_series_to_sitk, ants_to_monai


def crop_foreground_sitk(
    image: sitk.Image, threshold: float = 0.0, margin: int = 0
) -> sitk.Image:
    """
    Crops empty background (e.g. air) from a SimpleITK image by finding the
    bounding box of voxels above `threshold` and cropping to that region.
    Mirrors MONAI's CropForeground default behavior (img > 0).
    """
    mask = sitk.BinaryThreshold(
        image,
        lowerThreshold=threshold,
        upperThreshold=1e9,
        insideValue=1,
        outsideValue=0,
    )

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        return image  # nothing above threshold — return unchanged

    bbox = stats.GetBoundingBox(1)  # (x, y, z, size_x, size_y, size_z) in 3D
    ndim = image.GetDimension()
    start = list(bbox[:ndim])
    size = list(bbox[ndim:])

    img_size = image.GetSize()
    for i in range(ndim):
        start[i] = max(0, start[i] - margin)
        end = min(img_size[i], start[i] + size[i] + 2 * margin)
        size[i] = end - start[i]

    return sitk.RegionOfInterest(image, size=size, index=start)


def skullstrip(
    input_path: str,
    output_path: str,
    gpu: bool = True,
    image: str = "freesurfer/synthstrip:1.8-gpu",
):

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    input_dir = input_path.parent
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["docker", "run", "--rm"]
    if gpu:
        cmd += ["--gpus", "all"]

    cmd += [
        "-v",
        f"{input_dir}:/input",
        "-v",
        f"{output_dir}:/output",
        image,
        "-i",
        f"/input/{input_path.name}",
        "-o",
        f"/output/{output_path.name}",
    ]

    if gpu:
        cmd += ["-g"]

    subprocess.run(cmd, check=True)


def n4_correct(image: ANTsImage) -> ANTsImage:
    return ants.n4_bias_field_correction(image=image, shrink_factor=2)


def ants_atropos(input_img: ANTsImage) -> ANTsImage:
    # 1. Load skull-stripped image and brain mask
    mask = ants.threshold_image(input_img, low_thresh=1e-5, high_thresh=float("inf"))

    # 2. Run Atropos (k=3 for CSF, GM, WM)
    # 'PriorIntensityGMM' or 'Socrates' with MRF weight provides FAST-equivalent behavior
    segmentation = ants.atropos(
        a=input_img,
        x=mask,
        i="KMeans[3]",  # Initialization (or pass tissue prior images)
        m="[0.2,1x1x1]",  # MRF smoothness weight and radius (spatial prior)
        c="[5,0.0001]",  # 5 iterations max or convergence threshold
        verbose=0,
    )

    return segmentation["segmentation"]


# def percentile_clip(metatensor: MetaTensor):
#     """
#     Clips the tensor to the 1st and 99th percentile values to reduce noise.
#     """
#     array = metatensor.array
#     lower = np.percentile(array, 1)
#     upper = np.percentile(array, 99)
#     metatensor.array = np.clip(array, lower, upper)
#     return metatensor


def delete_extra_files(dir: str):
    """
    Deletes unneeded extra files in the directory to save space.
    """
    needed_files = ["img_seg.nii.gz", "img.nii.gz"]

    for file in os.scandir(dir):
        path = file.path
        if file.name not in needed_files:
            os.remove(path)


def preprocess_adni_pipeline(
    dicom_dir: str,
    out_dir: str,
    output_size: int,
    gpu: bool = False,
    target_spacing: tuple = (1.0, 1.0, 1.0),
):
    """
    Full pipeline:
    DCM -> N4 Bias Correction -> Skull Stripping -> 1mm Resampling (RAS) -> Intensity Percentile Clip -> Z-Score Normalization -> NIfTI
    """

    img_path = os.path.join(out_dir, "img.nii.gz")
    seg_path = os.path.join(out_dir, "img_seg.nii.gz")

    raw = dcm_series_to_sitk(dicom_dir)
    cropped = crop_foreground_sitk(raw)

    sitk.WriteImage(cropped, img_path)

    # ants.image_write(/raw, img_path)

    skullstrip(img_path, img_path, gpu=gpu)

    skull_stripped = ants.image_read(img_path)

    n4_corrected = n4_correct(skull_stripped)
    segmented = ants_atropos(n4_corrected)

    image = ants_to_monai(n4_corrected)
    label = ants_to_monai(segmented)

    dct = {"image": image, "label": label}

    transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
            NormalizeIntensityd(keys=["image"]),  # Z-score normalization
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(output_size, output_size, output_size),
            ),
        ]
    )

    out = transforms(dct)

    img = out["image"]
    seg = out["label"]

    writer = NibabelWriter()

    writer.set_data_array(img, channel_dim=0)
    writer.set_metadata({"affine": img.affine})
    writer.write(img_path)

    writer.set_data_array(seg, channel_dim=0)
    writer.set_metadata({"affine": seg.affine})
    writer.write(seg_path)

    delete_extra_files(out_dir)
    # Remove unneeded files


def list_usable_dcm_dirs(
    adni_root: str, mriqc_csv: str, min_slices: int = 20
) -> pd.DataFrame:
    adni_path = Path(adni_root)

    df_qc = pd.read_csv(mriqc_csv, low_memory=False)

    # Only want images with T1-weighted 3D scans
    passed_df = df_qc[
        (df_qc["SeriesType"] == "T1w") & (df_qc["AcquisitionType"] == "3D")
    ]

    id_col = "image_id"
    approved_ids = set(passed_df[id_col].dropna().astype(str))

    usable = []
    for root, _, files in os.walk(adni_path):

        dcm_count = sum(1 for f in files if f.lower().endswith(".dcm"))
        if dcm_count < min_slices:
            continue

        folder = Path(root)
        numeric_id = folder.name.lstrip("I")

        if numeric_id in approved_ids:
            usable.append(
                {
                    "subject_id": folder.parents[2].name,
                    "sequence": folder.parents[1].name,
                    "image_id": numeric_id,
                    "n_slices": dcm_count,
                    "path": str(folder.resolve()),
                }
            )

    return usable


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-dir",
        help="Input directory.",
        type=str,
        default="./data/raw/ADNI",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory.",
        type=str,
        default="./data/images",
    )
    parser.add_argument(
        "-q",
        "--qc",
        help="MRI quality control file path.",
        type=str,
        default="./data/tables/MRIQC.csv",
    )
    parser.add_argument(
        "-t",
        "--threads",
        help="Number of threads for segmentation algorithm.",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-n",
        "--max-images",
        help="Maximum images to process.",
        type=int,
        required=False,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--output-size",
        help="Output image size. Must be a multiple of 16.",
        type=int,
        default=256,
    )
    parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
    parser.add_argument(
        "-r",
        "--resume",
        help="Whether to resume from when the script crashed/terminated. \
                              Assumes that -n and -s parameters stay the same between runs.",
        action="store_true",
    )
    parser.add_argument(
        "-g", "--gpu", help="Run skullstrip on gpu.", action="store_true"
    )

    args = parser.parse_args()

    seed = args.seed
    threads = args.threads
    adni_root = args.input_dir
    out_dir = args.output_dir
    qc_file = args.qc
    max_images = args.max_images
    resume = args.resume
    out_size = args.output_size
    gpu = args.gpu

    # Argument validation
    assert out_size % 16 == 0, "Output size must be a multiple of 16!"

    # Seats number of threads to use for segmentation
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)

    usable_dirs = list_usable_dcm_dirs(
        adni_root=adni_root,
        mriqc_csv=qc_file,
        min_slices=20,  # Full 3D T1 acquisitions typically have 160-220 slices
    )

    # Export to DataFrame for processing pipelines

    print(f"\nFound {len(usable_dirs)} usable scan series.")

    raw_images: DataFrame = pd.DataFrame(usable_dirs)
    # Shuffle and retain a maximum of max_images images
    raw_images = raw_images.sample(frac=1, random_state=seed).head(max_images)

    total_images = len(raw_images)

    processed_images = set()

    if resume:
        for subject in os.scandir(out_dir):
            for image in os.scandir(subject.path):
                files = os.listdir(image.path)
                if "img.nii.gz" in files and "img_seg.nii.gz" in files:
                    processed_images.add(image.name)
                if len(files) > 2:
                    delete_extra_files(image.path)

    last_index = -1

    for i, entry in raw_images.iterrows():
        image_id = entry["image_id"]
        if image_id in processed_images:
            last_index = i

    raw_images = raw_images.iloc[last_index + 1 :]

    print(len(raw_images))

    for _, row in tqdm(
        raw_images.iterrows(), initial=last_index + 1, leave=True, total=total_images
    ):
        path = row["path"]
        subject = row["subject_id"]
        image_id = row["image_id"]

        img_out_dir = os.path.join(out_dir, subject, image_id)

        # print(out_dir)

        os.makedirs(img_out_dir, exist_ok=True)

        preprocess_adni_pipeline(path, img_out_dir, out_size, gpu)

    # input_dcm_folder = "path/to/ADNI/002_S_0295/MPRAGE/2006-04-18_.../S13408"
    # output_nii = "path/to/ADNI_clean/002_S_0295_MPRAGE_preprocessed.nii.gz"
    # preprocess_adni_pipeline(input_dcm_folder, output_nii)
    #


if __name__ == "__main__":
    main()
