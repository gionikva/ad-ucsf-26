"""Generate Mayo-filtered MRI–ADAS-Cog11 pairs from CSV metadata."""
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

# ADAS-Cog11 excludes delayed recall (Q4SCORE) and number
# cancellation (Q13SCORE). The output q1–q11 follows this order.
SOURCE_SCORES = [
    "Q1SCORE",
    "Q2SCORE",
    "Q3SCORE",
    "Q5SCORE",
    "Q6SCORE",
    "Q7SCORE",
    "Q8SCORE",
    "Q9SCORE",
    "Q10SCORE",
    "Q11SCORE",
    "Q12SCORE",
]

SCORE_LIMITS = [10, 5, 5, 5, 5, 8, 12, 5, 5, 5, 5]
OUTPUT_SCORES = [f"q{i}" for i in range(1, 12)]

OUTPUT_COLUMNS = [
    "subject_id",
    "image_id",
    "image_date",
    "score_date",
] + OUTPUT_SCORES


def normalize_text(values):
    """Clean a pandas Series and standardize its missing values."""
    return (
        values.astype("string")
        .str.strip()
        .replace(
            # Match blanks and missing-value codes, ignoring capitalization.
            r"(?i)^(?:|na|n/a|nan|null|-1|-4)$",
            pd.NA,
            regex=True,
        )
    )

def read_table(path, required_columns):
    """Read a CSV and report missing headers."""
    table = pd.read_csv(path, dtype="string")
    missing = set(required_columns) - set(table.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    return table

def normalize_image_ids(values):
    """Make image IDs such as I12345 and 12345 comparable."""
    return (
        normalize_text(values)
        .str.replace(r"^[Ii]", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )


def add_dimensions(mayo_data, dimensions_path):
    """Attach missing spatial dimensions using image IDs."""
    mayo = mayo_data.copy()
    mayo["image_id"] = normalize_image_ids(mayo["image_id"])
    missing = [c for c in ["Rows", "Columns", "SlicesPerVolume"]
               if c not in mayo.columns]
    if not missing:
        return mayo
    if dimensions_path is None:
        raise ValueError(
            f"Mayo metadata lacks {missing}. Supply --dimensions-table "
            "with image_id and the missing spatial dimensions."
        )
    dimensions = read_table(dimensions_path, ["image_id"] + missing)
    dimensions = dimensions[["image_id"] + missing].copy()
    dimensions["image_id"] = normalize_image_ids(dimensions["image_id"])
    dimensions = dimensions.dropna(subset=["image_id"]).drop_duplicates()
    if dimensions["image_id"].duplicated().any():
        raise ValueError("Conflicting dimensions for the same image_id.")
    return mayo.merge(dimensions, on="image_id", how="left",
                      validate="many_to_one")


def match_mri_adas11(
    mri_data,
    mayo_data,
    clinical_data,
    *,
    min_slices=100,
    max_gap_days=90,
    n_images=10000,
    seed=42,
):
    """Return sampled, unique MRI IDs with complete ADAS11 subscores."""
    if min_slices < 1:
        raise ValueError("min_slices must be at least 1.")
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be nonnegative.")
    if n_images < 0:
        raise ValueError("n_images must be nonnegative.")

    # Filter the Mayo inventory before considering clinical matches.
    # A missing or invalid slice count fails the filter.
    mayo = mayo_data.copy()
    mayo["image_id"] = normalize_image_ids(mayo["image_id"])
    
    # Require every spatial dimension to meet the minimum.
    dimensions = mayo[
        ["Rows", "Columns", "SlicesPerVolume"]
    ].apply(pd.to_numeric, errors="coerce")

    large_enough = (
        dimensions.ge(min_slices)
        .fillna(False)
        .all(axis=1)
    )

    eligible_ids = mayo.loc[
        normalize_text(mayo["SeriesType"]).str.casefold().eq("t1w")
        & normalize_text(mayo["AcquisitionType"]).str.upper().eq("3D")
        & large_enough,
        "image_id",
    ].dropna()

    scans = mri_data[
        ["subject_id", "image_id", "image_date"]
    ].copy()

    scans["image_id"] = normalize_image_ids(scans["image_id"])
    scans["subject_id"] = normalize_text(
        scans["subject_id"]
    ).str.upper()

    scans["image_date"] = pd.to_datetime(
        scans["image_date"],
        format="mixed",
        errors="coerce",
    ).dt.normalize()

    # Restrict the MRI table to eligible Mayo image IDs.
    scans = (
        scans.loc[scans["image_id"].isin(eligible_ids)]
        .dropna()
        .drop_duplicates()
    )

    # Identical metadata rows are collapsed above. Conflicting metadata
    # for one image ID needs correction before matching.
    if scans["image_id"].duplicated().any():
        raise ValueError(
            "MRI metadata contains conflicting rows for an image_id."
        )

    print(f"Unique MRI IDs passing the Mayo filters: {len(scans):,}")

    scores = clinical_data[
        ["PTID", "VISDATE"] + SOURCE_SCORES
    ].copy()

    scores = scores.rename(
        columns={
            "PTID": "subject_id",
            "VISDATE": "score_date",
        }
    )

    scores["subject_id"] = normalize_text(
        scores["subject_id"]
    ).str.upper()

    scores["score_date"] = pd.to_datetime(
        scores["score_date"],
        format="mixed",
        errors="coerce",
    ).dt.normalize()

    # Require all 11 component scores. Negative missing-value codes
    # and out-of-range scores become missing.
    for column, maximum in zip(SOURCE_SCORES, SCORE_LIMITS):
        values = pd.to_numeric(scores[column], errors="coerce")
        scores[column] = values.where(values.between(0, maximum))

    scores = scores.dropna().drop_duplicates()

    # Match each scan only to assessments from the same participant.
    candidates = scans.merge(
        scores,
        on="subject_id",
        how="inner",
    )

    # Use the absolute gap so assessments before or after the scan
    # can qualify within the configured window.
    candidates["_gap_days"] = (
        candidates["image_date"] - candidates["score_date"]
    ).abs().dt.days

    candidates = candidates.loc[
        candidates["_gap_days"].le(max_gap_days)
    ].copy()

    closest_gap = candidates.groupby(
        "image_id"
    )["_gap_days"].transform("min")

    nearest = candidates.loc[
        candidates["_gap_days"].eq(closest_gap)
    ].copy()

    # Equal-distance assessments or conflicting scores on the closest
    # date are ambiguous. Exclude those image IDs.
    ambiguous = nearest["image_id"].duplicated(keep=False)
    ambiguous_count = nearest.loc[
        ambiguous, "image_id"
    ].nunique()

    nearest = nearest.loc[~ambiguous].copy()

    print(f"Image IDs excluded for ambiguous matches: {ambiguous_count:,}")

    # Rename source subscores to the requested output q1–q11.
    nearest = nearest.rename(
        columns=dict(zip(SOURCE_SCORES, OUTPUT_SCORES))
    )

    pairs = (
        nearest[OUTPUT_COLUMNS]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    print(f"Eligible MRI–ADAS11 pairs before sampling: {len(pairs):,}")

    # Sample after matching so each selected image has complete targets.
    # n_images=0 keeps the full eligible list.
    if n_images > 0:
        if len(pairs) > n_images:
            pairs = pairs.sample(
                n=n_images,
                random_state=seed,
            )
        elif len(pairs) < n_images:
            print(
                f"Requested {n_images:,} images; "
                f"returning all {len(pairs):,} eligible images."
            )

    return pairs.sort_values(
        ["subject_id", "image_date", "image_id"]
    ).reset_index(drop=True)


def main():
    """Load inputs, generate pairs, and save the output CSV."""
    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        "-m",
        "--mri-table",
        type=Path,
        default=Path(
            "data/tables/All_Subjects_Key_MRI_13Nov2026.csv"
        ),
    )
    parser.add_argument(
        "-q",
        "--qc",
        type=Path,
        default=Path("data/tables/MRIQC.csv"),
    )
    parser.add_argument(
        "-c",
        "--clinical",
        type=Path,
        default=Path("data/tables/ADAS.csv"),
    )
    parser.add_argument(
        "--dimensions-table",
        type=Path,
        help="CSV with image_id and spatial dimensions missing from Mayo.",
    )
    parser.add_argument(
        "--min-slices",
        type=int,
        default=100,
        help="Minimum voxel count required in each spatial dimension.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=90,
        help="Maximum MRI-to-assessment date difference.",
    )
    parser.add_argument(
        "-n",
        "--n-images",
        type=int,
        default=10000,
        help="Maximum image count; use 0 for all eligible images.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("MRI_ADAS11_pairs.csv"),
    )

    args = parser.parse_args()

    try:
        mri_table = read_table(
            args.mri_table,
            ["subject_id", "image_id", "image_date"],
        )
        mayo_table = read_table(
            args.qc,
            [
                "image_id",
                "SeriesType",
                "AcquisitionType",
                "SlicesPerVolume",
            ],
        )
        mayo_table = add_dimensions(mayo_table, args.dimensions_table)
        clinical_table = read_table(
            args.clinical,
            ["PTID", "VISDATE"] + SOURCE_SCORES,
        )

        pairs = match_mri_adas11(
            mri_table,
            mayo_table,
            clinical_table,
            min_slices=args.min_slices,
            max_gap_days=args.max_gap_days,
            n_images=args.n_images,
            seed=args.seed,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(
        args.output,
        index=False,
        date_format="%Y-%m-%d",
    )

    print(f"\nSaved {len(pairs):,} unique image IDs to {args.output}")
    print(f"Unique participants: {pairs['subject_id'].nunique():,}")


if __name__ == "__main__":
    main()
