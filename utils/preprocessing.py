import io
import ants
import torch
import numpy as np
from monai.data.meta_tensor import MetaTensor
import nibabel as nib
import SimpleITK as sitk

def sitk_to_monai(image: sitk.Image, device="cpu") -> MetaTensor:
    """
    Converts a SimpleITK image to a Monai metatensor for transforms.
    """
    # 1. SimpleITK array is (Z, Y, X) -> transpose to (X, Y, Z) and add channel -> (1, X, Y, Z)
    data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
    tensor = torch.from_numpy(data).unsqueeze(0)
    tensor.to(device)

    # 2. Build 4x4 Affine matrix
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())
    direction = np.array(image.GetDirection()).reshape(3, 3)

    affine = np.eye(4, dtype=np.float32)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin

    return MetaTensor(tensor, affine=torch.from_numpy(affine))

def monai_to_ants(meta_tensor):
    """Converts a MONAI MetaTensor to an ANTsImage directly in memory."""
    # 1. Extract array data (remove channel dimension if present)
   
    data_tensor = meta_tensor.squeeze()

    data_np = data_tensor.detach().cpu().numpy().astype(np.float32)

    # 2. Extract 4x4 affine matrix from MetaTensor
    affine = meta_tensor.affine.detach().cpu().numpy()

    # 3. Convert affine from RAS (NIfTI/MONAI) to LPS (ITK/ANTs)
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_lps = ras_to_lps @ affine

    # 4. Decompose the LPS affine into Origin, Spacing, and Direction
    origin = tuple(affine_lps[:3, 3])

    linear_part = affine_lps[:3, :3]
    spacing = tuple(np.linalg.norm(linear_part, axis=0))

    # Normalize columns to obtain the unit direction cosine matrix
    direction = linear_part / spacing

    # 5. Build the ANTsImage
    ants_img = ants.from_numpy(
        data=data_np, origin=origin, spacing=spacing, direction=direction
    )

    return ants_img

def ants_to_monai(
    ants_img: ants.ANTsImage,
    device="cpu"
) -> MetaTensor:
    """Converts an ANTsImage to a MONAI MetaTensor directly in memory,

    correctly converting LPS (ITK/ANTs) space to RAS (MONAI/NIfTI) space.
    """
    data = ants_img.numpy()
    tensor = torch.from_numpy(data)
    tensor.to(device)
    
    # Extract array data (ANTs uses (X, Y, Z))
    spacing = np.array(ants_img.spacing, dtype=np.float32)
    origin = np.array(ants_img.origin, dtype=np.float32)
    direction = np.array(ants_img.direction, dtype=np.float32)

    affine_lps = np.eye(4, dtype=np.float32)
    affine_lps[:3, :3] = direction @ np.diag(spacing)
    affine_lps[:3, 3] = origin

    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
    affine_ras = lps_to_ras @ affine_lps

    # 5. Build and return MetaTensor
    return MetaTensor(tensor, affine=torch.from_numpy(affine_ras))