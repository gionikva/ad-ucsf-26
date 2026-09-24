import os
from os import listdir
from monai.data.meta_tensor import MetaTensor
import torch
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import itk
import monai
from monai.transforms import (
    Compose,
    Spacing,
    Orientation,
    ScaleIntensity,
    NormalizeIntensity)
from monai.data import NibabelWriter
import nibabel as nib
import pandas as pd


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


def sitk_to_metatensor(image: sitk.Image) -> tio.Subject:
    """
    Converts a SimpleITK image to a Monai metatensor for transforms.
    """
    # 1. SimpleITK array is (Z, Y, X) -> transpose to (X, Y, Z) and add channel -> (1, X, Y, Z)
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    tensor = torch.from_numpy(data).unsqueeze(0)

    # 2. Build 4x4 Affine matrix
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin

    return MetaTensor(tensor, affine=torch.from_numpy(affine))


def preprocess_adni_pipeline(
    dicom_dir: str, out_dir: str, image_id: str, target_spacing: tuple = (1.0, 1.0, 1.0)
):
    """
    Full pipeline:
    DCM -> N4 Bias Correction -> Skull Stripping -> 1mm Resampling (RAS) -> Intensity Percentile Clip -> Z-Score Normalization -> NIfTI
    """
    sitk_img = dcm_series_to_sitk(dicom_dir)
    n4_corrected, initial_mask = apply_n4_bias_field_correction(sitk_img)
    brain_extracted = skull_strip(n4_corrected, initial_mask)
    tensor = sitk_to_metatensor(brain_extracted)

    normalize = Compose(
        [
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=0.01,
                           maxv=0.99),
            NormalizeIntensity(),
        ]
    )
    
    normalized = normalize(tensor)

    writer = NibabelWriter()

    img = normalized.squeeze(0) if normalized.ndim == 4 else normalized
    
    writer.set_data_array(img, channel_dim=None)
    writer.set_metadata({"affine": img.affine})
    writer.write(os.path.join(out_dir, f"{image_id}.nii.gz"))


def list_usable_dcm_dirs(
    adni_root: str, mriqc_csv: str, min_slices: int = 100
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
    adni_raw = "./data/raw/ADNI"
    qc_file = "./data/raw/MRIQC.csv"  # Set to None if you don't have it downloaded yet

    usable_dirs = list_usable_dcm_dirs(
        adni_root=adni_raw,
        mriqc_csv=qc_file,
        min_slices=100,  # Full 3D T1 acquisitions typically have 160-220 slices
    )

    # Export to DataFrame for processing pipelines
    df_usable = pd.DataFrame(usable_dirs)
    print(f"\nFound {len(df_usable)} usable scan series.")
    print(df_usable[["subject_id", "image_id", "path"]].head())

    root = "./data/images"

    for series in tqdm(usable_dirs):
        path = series["path"]
        subject = series["subject_id"]
        image_id = series["image_id"]

        out_dir = os.path.join(root, subject)

        os.makedirs(out_dir, exist_ok=True)

        preprocess_adni_pipeline(path, out_dir, image_id)

    # input_dcm_folder = "path/to/ADNI/002_S_0295/MPRAGE/2006-04-18_.../S13408"
    # output_nii = "path/to/ADNI_clean/002_S_0295_MPRAGE_preprocessed.nii.gz"
    # preprocess_adni_pipeline(input_dcm_folder, output_nii)
    #


if __name__ == "__main__":
    main()
