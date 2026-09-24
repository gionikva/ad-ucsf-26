from typing import Tuple

import pandas as pd
import numpy as np


def preprocess(mris: pd.DataFrame, pets: pd.DataFrame, pet_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, np.array]:
    pets["amyloid_pet"] = pets["amyloid_pet"] == "Y"
    pets["tau_pet"] = pets["tau_pet"] == "Y"
    
    subjects_pet = pets[pets[pet_type]]["subject_id"].unique()
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
  
   
    # print(pets["amyloid_pet"][:20])
    
  
    
    
    def calc_data_availability(mris, pets, pet_type: str):
        assert(pet_type in ["amyloid_pet", "tau_pet"])
        
        mris, pets, subjects = preprocess(mris, pets, pet_type)
        
        mins = []
        number_of_0s = 0
        number_less_1wk = 0
        number_less_1mo = 0
                
        for sid in subjects:
            subject_mri_dates = mris[mris["subject_id"] == sid]["image_date"]
            subject_pet_dates = pets[pets["subject_id"] == sid]["image_date"]
            
            min_deltas = []
            
            for pet_date in subject_pet_dates:
                deltas = np.abs(subject_mri_dates - pet_date)
                min_deltas.append(np.min(deltas))

            minimum = min(min_deltas)
        
            mins.append(minimum)
            if minimum == np.timedelta64(0, "D"):
                number_of_0s += 1
            if minimum <= np.timedelta64(14, "D"):
                number_less_1wk += 1
            if minimum <= np.timedelta64(30, "D"):
                number_less_1mo += 1
        
        # print(f"Min time difference between mris and PET scan for subject{sid}: {min(min_deltas)}")
        
        mins = np.array(mins, np.timedelta64)
        
        median_min_time_diff = np.median(mins) / np.timedelta64(1, "D")
        
        prop_0 = number_of_0s / total_subjects
        prop_less_1wk = number_less_1wk / total_subjects
        prop_less_1mo = number_less_1mo / total_subjects
        
        print(f"Median minimum time difference between MRIs and PET scans ({pet_type}): {median_min_time_diff}")
        print(f"Proportion of min time diffs | 0 days: {prop_0:.4f} | <= 1 wk: {prop_less_1wk:.4f} | <= 1 mo: {prop_less_1mo:.4f}")
        
    
    calc_data_availability(mris.copy(), pets.copy(), "amyloid_pet")
    print()
    calc_data_availability(mris, pets, "tau_pet")
    
    # print(len(mris), len(pets))
    

def main():
    compare_scan_times()
    

if __name__ == "__main__":
    main()


