from typing import Tuple

import pandas as pd
import numpy as np


def preprocess(mris: pd.DataFrame, pets: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    
    pets["amyloid_pet"] = pets["amyloid_pet"] == "Y"
    pets["tau_pet"] = pets["tau_pet"] == "Y"
    
    # The following code generates the subjects in pet that have both amyloid and tau pet available
    subject_summary = pets.groupby('subject_id')[['tau_pet', 'amyloid_pet']].any()
    subjects_pet = subject_summary[subject_summary['tau_pet'] & subject_summary['amyloid_pet']].index    
    
    subjects_mri = mris["subject_id"].unique()
    
    subjects = np.intersect1d(subjects_pet, subjects_mri)
    
    mris = mris[mris["subject_id"].isin(subjects)]
    pets = pets[pets["subject_id"].isin(subjects)]
    
    mris["image_date"] = pd.to_datetime(mris["image_date"])
    pets["image_date"] = pd.to_datetime(pets["image_date"])
    
    
    
    return mris, pets, subjects


def compare_scan_times():
    mris = pd.read_csv("./data/tables/All_Subjects_Key_MRI_10Sep2026.csv")
    pets = pd.read_csv("./data/tables/All_Subjects_Key_PET_10Sep2026.csv")
    
  
    total_subjects = len(mris["subject_id"].unique())

  
    mris, pets, subjects = preprocess(mris, pets)
    # print(pets["amyloid_pet"][:20])
    
    # assert(pet_type in ["amyloid_pet", "tau_pet"])
    
    pets_amy = pets[pets["amyloid_pet"]]
    pets_tau = pets[pets["tau_pet"]]
    
    mins = []
    number_of_0s = 0
    number_less_1wk = 0
    number_less_1mo = 0
    
    for sid in subjects:
        subject_mri_dates = mris[mris["subject_id"] == sid]["image_date"]
        amy_pet_dates = pets_amy[pets_amy["subject_id"] == sid]["image_date"]
        tau_pet_dates = pets_tau[pets_tau["subject_id"] == sid]["image_date"]
        
        min_deltas_amy = []
        min_deltas_tau = []
        
        for pet_date in amy_pet_dates:
            deltas = np.abs(subject_mri_dates - pet_date)
            min_deltas_amy.append(np.min(deltas))
        
        for pet_date in tau_pet_dates:
            deltas = np.abs(subject_mri_dates - pet_date)
            min_deltas_tau.append(np.min(deltas))

        minimum = max(min(min_deltas_amy), min(min_deltas_tau))
    
        mins.append(minimum)
        if minimum == np.timedelta64(0, "D"):
            number_of_0s += 1
        if minimum <= np.timedelta64(14, "D"):
            number_less_1wk += 1
        if minimum <= np.timedelta64(30, "D"):
            number_less_1mo += 1
    
    # print(f"Min time difference between mris and PET scan for subject{sid}: {min(min_deltas)}")
    
    mins = np.array(mins, np.timedelta64)
    
    median_min_time_diff = np.median(mins) / np.timedelta64(1, 'D')
    
    prop_0 = number_of_0s / total_subjects
    prop_less_1wk = number_less_1wk / total_subjects
    prop_less_1mo = number_less_1mo / total_subjects
    
    print(f"Median minimum time difference between MRIs and PET scans for both amyloid and tau pet: {median_min_time_diff}")
    print(f"Proportion of min time diffs | 0 days: {prop_0:.4f} | <= 1 wk: {prop_less_1wk:.4f} | <= 1mo: {prop_less_1mo:.4f}")
    
    
    
    print(len(mris), len(pets))
    

def main():
    compare_scan_times()
    

if __name__ == "__main__":
    main()


