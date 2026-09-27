import io
import ants
import torch
import numpy as np
from monai.data.meta_tensor import MetaTensor
import nibabel as nib
import SimpleITK as sitk


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

def sitk_to_monai(image: sitk.Image, device="cpu") -> MetaTensor:
    """
    Converts a SimpleITK image to a Monai metatensor for transforms.
    """
    # 1. SimpleITK array is (Z, Y, X) -> transpose to (X, Y, Z) and add channel -> (1, X, Y, Z)
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    tensor = torch.from_numpy(data).unsqueeze(0)
    tensor = tensor.to(device)


    print(np.array(image.GetSpacing()))
    print(np.array(image.GetOrigin()))
    print(np.array(image.GetDirection()))

    raise Exception("stopping")

    
    # 2. Build 4x4 Affine matrix
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = direction @ spacing
    affine[:3, 3] = origin

#     return MetaTensor(tensor, affine=torch.from_numpy(affine))
def ants_to_monai(ants_img: ants.ANTsImage, device="cpu") -> MetaTensor:
    """
    Converts an ANTsImage to a MONAI MetaTensor directly in memory,
    correctly converting LPS (ITK/ANTs) space to RAS (MONAI/NIfTI) space.
    """
    data = ants_img.numpy()  # (X, Y, Z)
    tensor = torch.from_numpy(data).to(device)

    spacing = np.array(ants_img.spacing, dtype=np.float64)
    origin = np.array(ants_img.origin, dtype=np.float64)
    direction = np.array(ants_img.direction, dtype=np.float64).reshape(3, 3)

    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps

    return MetaTensor(tensor, affine=torch.from_numpy(affine_ras))