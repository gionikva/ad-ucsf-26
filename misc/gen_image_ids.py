from random import shuffle
from argparse import ArgumentParser
import pandas as pd
import numpy as np


def list_usable_dcm_dirs(mriqc_csv: str, min_slices=20) -> pd.Series:
    df_qc = pd.read_csv(mriqc_csv, low_memory=False)

    # Only want images with T1-weighted 3D scans
    passed_df = df_qc[
        (df_qc["SeriesType"] == "T1w")
        & (df_qc["AcquisitionType"] == "3D")
        & (df_qc["SlicesPerVolume"] >= min_slices)
    ]

    id_col = "image_id"
    usable = pd.Series(passed_df[id_col].dropna().astype(str))

    return usable

    # usable = []
    # for root, _, files in os.walk(adni_path):
    #     dcm_count = sum(1 for f in files if f.lower().endswith(".dcm"))
    #     if dcm_count < min_slices:
    #         continue

    #     folder = Path(root)
    #     numeric_id = folder.name.lstrip("I")

    #     if numeric_id in approved_ids:
    #         usable.append(
    #             {
    #                 "subject_id": folder.parents[2].name,
    #                 "sequence": folder.parents[1].name,
    #                 "image_id": numeric_id,
    #                 "n_slices": dcm_count,
    #                 "path": str(folder.resolve()),
    #             }
    #         )

    # return usable


# def _per_subject():
#     parser = ArgumentParser()

#     parser.add_argument("-m", "--mri-table", type=str, default="./data/tables/All_Subjects_Key_MRI_10Sep2026.csv")
#     parser.add_argument("-q", "--qc", type=str, default="./data/tables/MRIQC.csv")
#     parser.add_argument("-n", "--n-subjects", type=int, required=True)
#     parser.add_argument("-p", "--images-per-subject", type=int, default=4)
#     parser.add_argument("-s", "--seed", type=int, default=42)

#     args = parser.parse_args()

#     mris_path = args.mri_table
#     qc_path = args.qc
#     seed = args.seed
#     n_samples = args.n
#     n_img_per_subject = args.images_per_subject

#     mris = pd.read_csv(mris_path)
#     qc = pd.read_csv(qc_path)

#     subjects = pd.Series(mris["subject_id"].unique()).sample(
#         n=n_samples, random_state=seed
#     )

#     mris = mris[mris["subject_id"].isin(subjects)]
#     images = (mris.sample(frac=1, random_state=seed)
#                .groupby("subject_id").head(4)["image_id"]
#                .tolist())

#     print(images)
#     # .sample(n=n_img_per_subject, random_state=seed)
#     # images = grouped["imaged_id"].unique()

#     print(len(images))

#     with open("./image_sample.txt", 'w') as file:
#         file.write(",".join([str(img) for img in list(images)]))


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--mri-table",
        type=str,
        default="./data/tables/All_Subjects_Key_MRI_10Sep2026.csv",
    )
    parser.add_argument("-q", "--qc", type=str, default="./data/tables/MRIQC.csv")
    parser.add_argument("-n", "--n-images", type=int, required=True)
    parser.add_argument("-s", "--seed", type=int, default=42)

    args = parser.parse_args()

    mris_path = args.mri_table
    qc_path = args.qc
    seed = args.seed
    n_samples = args.n_images

    mris = pd.read_csv(mris_path)
    qc = pd.read_csv(qc_path)

    image_ids: pd.Series = list_usable_dcm_dirs(
        mriqc_csv=qc_path,
        min_slices=100,  # Full 3D T1 acquisitions typically have 160-220 slices
    )

    images = image_ids.sample(
        n=n_samples,
        random_state=seed
    )
    
    print(len(images))

    with open("./misc/images_sample.txt", "w") as file:
        file.write(",".join([str(img) for img in images]))


if __name__ == "__main__":
    main()
