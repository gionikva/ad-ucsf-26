from random import shuffle
from argparse import ArgumentParser
import pandas as pd


def main():
    parser = ArgumentParser()

    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-i", "--images-per-subject", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=42)

    args = parser.parse_args()

    seed = args.seed
    n_samples = args.n
    n_img_per_subject = args.images_per_subject

    mris = pd.read_csv("./data/tables/All_Subjects_Key_MRI_10Sep2026.csv")

    subjects = pd.Series(mris["subject_id"].unique()).sample(
        n=n_samples, random_state=seed
    )
    
    mris = mris[mris["subject_id"].isin(subjects)]
    images = (mris.sample(frac=1, random_state=seed)
               .groupby("subject_id").head(4)["image_id"]
               .tolist())
    
    print(images)
    # .sample(n=n_img_per_subject, random_state=seed)
    # images = grouped["imaged_id"].unique()
    
    print(len(images))

    with open("./image_sample.txt", 'w') as file:
        file.write(",".join([str(img) for img in list(images)]))


if __name__ == "__main__":
    main()
