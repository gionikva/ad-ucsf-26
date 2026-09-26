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
    Spacing,
    Orientation,
    ScaleIntensityRange,
    NormalizeIntensityd,
    CropForeground,
    LoadImaged,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
)
from monai.data import NibabelWriter
import nibabel as nib
import pandas as pd
from utils.wrappers.adni import get_image_dirs
from utils.preprocessing import (
    sitk_to_monai,
    monai_to_ants,
    ants_to_monai,
)
import random


def dcm_series_to_sitk(dicom_dir: str) -> sitk.Image:
    """
    Reads a directory of 2D .dcm slices and stacks them into a 3D SimpleITK Image,
    preserving coordinate spaces, origin, spacing, and directions.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise FileNotFoundError(f"No valid DICOM series found in: {dicom_dir}")

    # Load the first series found in the directory
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return sitk.Cast(image, sitk.sitkFloat32)


def apply_n4_bias_field_correction(image: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    """
    Computes an Otsu background mask and corrects RF inhomogeneity.
    Returns: (corrected_image, brain_mask)
    """
    # Generate initial foreground mask to guide N4
    mask = sitk.OtsuThreshold(image, 0, 1, 200)

    # Optional: shrink image for faster spline computation
    shrink_factor = [2, 2, 2]
    shrunk_image = sitk.Shrink(image, shrink_factor)
    shrunk_mask = sitk.Shrink(mask, shrink_factor)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrector.SetConvergenceThreshold(0.001)

    _ = corrector.Execute(shrunk_image, shrunk_mask)

    # Evaluate full-resolution bias field
    log_bias_field = corrector.GetLogBiasFieldAsImage(image)
    corrected_image = image / sitk.Exp(log_bias_field)

    return corrected_image, mask


def skull_strip(image: sitk.Image, mask: sitk.Image) -> sitk.Image:
    """
    Applies the binary mask to zero out background/skull non-brain voxels.
    """
    # Morphological opening and largest connected component to isolate cerebrum/cerebellum
    cleaned_mask = sitk.BinaryMorphologicalOpening(mask, (3, 3, 3))
    cleaned_mask = sitk.RelabelComponent(sitk.ConnectedComponent(cleaned_mask)) == 1

    # Zero-out voxels outside the mask
    masked_image = sitk.Mask(image, cleaned_mask, maskingValue=0.0)
    return masked_image


def percentile_clip(metatensor: MetaTensor):
    """
    Clips the tensor to the 1st and 99th percentile values to reduce noise.
    """
    array = metatensor.array
    lower = np.percentile(array, 1)
    upper = np.percentile(array, 99)
    metatensor.array = np.clip(array, lower, upper)
    return metatensor


# def run_fsl_fast(input_dir: str):
#     """Uses FSL-fast to segment brain into GM, WM, CSF."""
#     cmd = [
#         "fast",
#         "-t",
#         "1",  # T1-weighted
#         "-n",
#         "3",
#         "-N",
#         "-o",
#         os.path.join(input_dir, "img"),
#         os.path.join(input_dir, "temp.nii.gz")
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True)
#     if result.returncode != 0:
#         raise RuntimeError(f"FAST failed:\n{result.stderr}")


def ants_atropos(input_img) -> ANTsImage:
    # 1. Load skull-stripped image and brain mask
    mask = mask = ants.threshold_image(
        input_img, low_thresh=1e-5, high_thresh=float("inf")
    )

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

    # segmentation['segmentation'] -> Hard label mask (1=CSF, 2=GM, 3=WM)
    # segmentation['probabilityimages'] -> 4D array / list of posterior probability maps
    return segmentation["segmentation"]


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
    device="cpu",
    target_spacing: tuple = (1.0, 1.0, 1.0),
):
    """
    Full pipeline:
    DCM -> N4 Bias Correction -> Skull Stripping -> 1mm Resampling (RAS) -> Intensity Percentile Clip -> Z-Score Normalization -> NIfTI
    """
    sitk_img = dcm_series_to_sitk(dicom_dir)
    n4_corrected, initial_mask = apply_n4_bias_field_correction(sitk_img)
    brain_extracted = skull_strip(n4_corrected, initial_mask)
    tensor = sitk_to_monai(brain_extracted)
    tensor = percentile_clip(tensor)
    tensor.to(device)

    pre_fast = Compose(
        [
            CropForeground(),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientation(axcodes="RAS", labels=None),
        ]
    )

    normalized = pre_fast(tensor)
    
    img = normalized.squeeze(0) if normalized.ndim == 4 else normalized    
    seg = ants_to_monai(ants_atropos(monai_to_ants(img)), device)

    img_path = os.path.join(out_dir, f"img.nii.gz")
    seg_path = os.path.join(out_dir, f"img_seg.nii.gz")

    dct = {"image": img, "label": seg}

    final_transforms = Compose(
        [
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys=["image"]),  # Z-score normalization
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(output_size, output_size, output_size),
            ),
        ]
    )

    out = final_transforms(dct)
    
    img = out["image"].squeeze(0)
    seg = out["label"].squeeze(0)
    
    writer = NibabelWriter()

    writer.set_data_array(img, channel_dim=None)
    writer.set_metadata({"affine": img.affine})
    writer.write(img_path)

    writer.set_data_array(seg, channel_dim=None)
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

    args = parser.parse_args()

    seed = args.seed
    threads = args.threads
    adni_root = args.input_dir
    out_dir = args.output_dir
    qc_file = args.qc
    max_images = args.max_images
    resume = args.resume
    out_size = args.output_size

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

        preprocess_adni_pipeline(path, img_out_dir, out_size)

    # input_dcm_folder = "path/to/ADNI/002_S_0295/MPRAGE/2006-04-18_.../S13408"
    # output_nii = "path/to/ADNI_clean/002_S_0295_MPRAGE_preprocessed.nii.gz"
    # preprocess_adni_pipeline(input_dcm_folder, output_nii)
    #


if __name__ == "__main__":
    main()
