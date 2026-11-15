import pandas as pd


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


def match_adni_adas11(
    mri_data,
    clinical,
    diagnosis,
    *,
    mri_id_col="image_id",
    mri_ptid_col="subject_id",
    mri_date_col="image_date",
    diagnosis_date_col="EXAMDATE",
    diagnosis_value_col="DIAGNOSIS",
    max_gap_days=90,
):
    """Return one output row per MRI, with ADAS11 and diagnosis when available."""
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be nonnegative.")

    # Check the headers up front so a mismatched export gives a clear error.
    for name, table, required in (
        ("MRI metadata", mri_data, {mri_id_col, mri_ptid_col, mri_date_col}),
        ("ADAS11 data", clinical, {"PTID", "EXAMDATE", "ADAS11"}),
        ("Diagnosis data", diagnosis,
         {"PTID", diagnosis_date_col, diagnosis_value_col}),
    ):
        missing = required - set(table.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    # Work on copies. Reset the clinical index so clinical_row refers to
    # the zero-based row position in the clinical input.
    scans = mri_data.copy()
    visits = clinical.copy().reset_index(drop=True)
    diagnoses = diagnosis.copy()

    scans["_ptid"] = normalize_text(scans[mri_ptid_col]).str.upper()
    visits["_ptid"] = normalize_text(visits["PTID"]).str.upper()
    diagnoses["_ptid"] = normalize_text(diagnoses["PTID"]).str.upper()

    # Compare calendar dates. Unparseable values become missing dates.
    scans["_scan_date"] = pd.to_datetime(
        scans[mri_date_col], format="mixed", errors="coerce"
    ).dt.normalize()

    # In ADNIMERGE, EXAMDATE is the clinical visit date. It may differ
    # from the exact date the participant took the ADAS-Cog11 exam.
    visits["_visit_date"] = pd.to_datetime(
        visits["EXAMDATE"], format="mixed", errors="coerce"
    ).dt.normalize()

    diagnoses["_diagnosis_date"] = pd.to_datetime(
        diagnoses[diagnosis_date_col], format="mixed", errors="coerce"
    ).dt.normalize()

    # Keep valid ADAS11 totals, including zero, and exclude missing codes.
    totals = pd.to_numeric(visits["ADAS11"], errors="coerce")
    visits["ADAS11"] = totals.where(totals.between(0, 70))

    # Preserve diagnosis codes as supplied by the export. Use the ADNI
    # data dictionary for your release to interpret numeric codes.
    diagnoses["_diagnosis"] = normalize_text(diagnoses[diagnosis_value_col])

    # Build independent lookups. A diagnosis visit can be eligible even
    # when the participant has no ADAS11 score at that visit.
    score_rows = visits.dropna(subset=["_ptid", "_visit_date", "ADAS11"])
    diagnosis_rows = diagnoses.dropna(
        subset=["_ptid", "_diagnosis_date", "_diagnosis"]
    )
    scores_by_ptid = dict(tuple(score_rows.groupby("_ptid")))
    diagnoses_by_ptid = dict(tuple(diagnosis_rows.groupby("_ptid")))

    columns = [
        "mri_id", "PTID", "scan_date", "status", "reason",
        "clinical_row", "clinical_visit_date", "VISCODE", "ADAS11",
        "signed_visit_gap_days", "candidate_count", "match_method",
        "diagnosis_at_mri", "diagnosis_visit_date", "diagnosis_gap_days",
        "diagnosis_status",
    ]
    output = []

    for _, scan in scans.iterrows():
        ptid = scan["_ptid"]
        scan_date = scan["_scan_date"]

        result = dict.fromkeys(columns)
        result.update(
            mri_id=scan[mri_id_col],
            PTID=ptid,
            scan_date=scan_date,
            status="unmatched",
            candidate_count=0,
            diagnosis_status="unknown",
        )
        output.append(result)

        if pd.isna(ptid) or pd.isna(scan_date):
            result.update(
                status="review",
                reason="Missing participant ID or invalid MRI date",
            )
            continue

        participant_scores = scores_by_ptid.get(ptid)
        if participant_scores is None:
            result["reason"] = "No dated visit with a valid ADAS11 score"
            continue

        # Negative gaps mean the visit came before the scan. Positive
        # gaps mean it came after. Either direction can supply ADAS11.
        gaps = (participant_scores["_visit_date"] - scan_date).dt.days
        candidate_gaps = gaps.loc[gaps.abs() <= max_gap_days]
        result["candidate_count"] = len(candidate_gaps)

        if candidate_gaps.empty:
            result["reason"] = "No valid ADAS11 score within the date window"
            continue

        nearest_gap = candidate_gaps.abs().min()
        nearest_rows = candidate_gaps.index[candidate_gaps.abs() == nearest_gap]

        # Flag ties rather than choosing an assessment arbitrarily.
        if len(nearest_rows) != 1:
            result.update(
                status="review",
                reason="Multiple equally close ADAS11 records",
            )
            continue

        clinical_row = nearest_rows[0]
        assessment = participant_scores.loc[clinical_row]
        result.update(
            status="matched",
            clinical_row=int(clinical_row),
            clinical_visit_date=assessment["_visit_date"],
            VISCODE=assessment.get("VISCODE"),
            ADAS11=float(assessment["ADAS11"]),
            signed_visit_gap_days=int(gaps.loc[clinical_row]),
            match_method="same_day" if nearest_gap == 0 else "nearest_date",
        )

        # Diagnosis uses a different date rule: only records on or before
        # the scan are eligible. The ADAS11 window does not limit this lookup.
        participant_diagnoses = diagnoses_by_ptid.get(ptid)
        if participant_diagnoses is None:
            continue

        prior = participant_diagnoses.loc[
            participant_diagnoses["_diagnosis_date"] <= scan_date
        ]
        if prior.empty:
            continue

        latest_date = prior["_diagnosis_date"].max()
        latest_labels = prior.loc[
            prior["_diagnosis_date"] == latest_date, "_diagnosis"
        ].unique()

        result.update(
            diagnosis_visit_date=latest_date,
            diagnosis_gap_days=(scan_date - latest_date).days,
        )

        # Repeated copies of the same label agree. Different labels on
        # the latest eligible date leave the diagnosis blank for review.
        if len(latest_labels) == 1:
            result.update(
                diagnosis_at_mri=latest_labels[0],
                diagnosis_status="documented",
            )
        else:
            result["diagnosis_status"] = "conflicting"

    return pd.DataFrame(output, columns=columns)


if __name__ == "__main__":
    # Replace these filenames with your actual exports. Read identifiers
    # as strings so pandas preserves leading zeros.
    mri_data = pd.read_csv(
    "data/tables/All_Subjects_Key_MRI_13Nov2026.csv",
    dtype="string",
    )
    clinical_table = pd.read_csv("data/tables/ADAS.csv", dtype="string").rename(
        columns={"VISDATE": "EXAMDATE", "TOTSCORE": "ADAS11"}
    )
    diagnosis_table = pd.read_csv("data/tables/DXSUM.csv", dtype="string")

    matches = match_adni_adas11(
        mri_data,
        clinical_table,
        diagnosis_table,
        mri_id_col="image_id",
        mri_ptid_col="subject_id",
        mri_date_col="image_date",
        diagnosis_date_col="EXAMDATE",
        diagnosis_value_col="DIAGNOSIS",
        max_gap_days=90,
    )

    matches.to_csv("MRI_ADAS11_matches.csv", index=False)

    print("MRI matching results:")
    print(matches["status"].value_counts().to_string())

    print("\nDiagnosis status among matched MRI–ADAS11 pairs:")
    print(
        matches.loc[matches["status"] == "matched", "diagnosis_status"]
        .value_counts()
        .to_string()
    )

# Count MRI rows linked to the same ADAS-Cog11 assessment.
    matched = matches.loc[matches["status"].eq("matched")]
    scans_per_assessment = matched.groupby(["PTID", "clinical_row"]).size()
    shared = scans_per_assessment[scans_per_assessment > 1]
    scans_per_patient = matched.groupby("PTID")["mri_id"].nunique()
    patients_with_multiple_scans = scans_per_patient.gt(1).sum()

    print(f"\n{len(shared):,} ADAS-Cog11 assessments are linked to multiple MRI scans.")
    print(f"{patients_with_multiple_scans:,} patients have two or more MRI scans matched to ADAS-Cog11.\n")