# ad-ucsf-26
Alzheimer's disease UCSF/Berkeley/BU research project 2026.
## Acquiring the data
Visit the [loni website](https://ida.loni.usc.edu) and apply for access to ADNI from the "Studies you can apply to" panel on the right. 

Once access has been granted, navigate to the ARC Builder by clicking on the menu on the top right.

The ARC builder should open with the tables section. Navigate to the `"Downloads"` tab, go to `"Study Files"`, search for `"MAYO ADIR LAB MRI quality"`. Download the table and save it to `ad-ucsf-26/data/tables/MRIQC.csv`.

This file is used to programatically identify suitable structural MRI scans and generate a list of image IDs that you will later download.
 
To generate the image IDs, run `python misc/gen_image_ids.py`, optionally with the `-n` parameter (I recommend starting with 5000) to generate a list of image IDs. 

Then, go back to the ADNI ARC builder, click on the `+` (new filter) button in the bottom pane and choose `"choose images from image IDs that you provide"`. Then, paste in the content of `images_sample.txt`, click on `"Add Image IDs"` and then `"Done"`.

Afterward, navigate to `"Downloads" > "Images"` and download the files. Once the files have been downloaded, extract them all into the directory `ad-ucsf-26/data/raw/ADNI`. Please make sure that the `data/raw/ADNI` directory contains the subfolders corresponding to the study particpants.

## Preprocessing the data
First, install the dependencies by running `pip install -r requirements.txt`. 

To preprocess the data, run `python preprocess.py -d 192 --segment`. In case the script crashes or is interrupted, you may resume preprocessing with `python preprocess.py -d 192 -r`, though do note this functionality has not been extensively tested so I would use it with caution. In general, you may `python preprocess.py --help` for all available options.

## Segmentation pretraining
To train the (LightMedSeg) backbone on the BM/WM/CSF segmentation pretraining task, run 

```python train_seg.py -o weights/<placeholder> --batch-size <BATCH_SIZE> -e 100 -s medium -a --deep-supervision```.


By default, this will *not* downsample the input `192^3` volumes to avoid destroying fine detail, which might result in out of memory errors and very slow training. If this does occur, simply add the `-d` or `--downsample` parameter to enable downsampling in the model. 

## MRI–ADAS-Cog11 pairing and slice-count filtering
 
Run `MRI_ADAS-Cog11_Matching.py` from the repository root to generate up to 10,000 unique image IDs paired with complete ADAS-Cog11 subscores. The generator reads CSV metadata only. Prepare these inputs:
 
- `data/tables/All_Subjects_Key_MRI_13Nov2026.csv` with `image_id`, `subject_id`, and `image_date`.
- `data/tables/MRIQC.csv`, the MAYO ADIR LAB MRI inventory, with `image_id`, `SeriesType`, `AcquisitionType`, and `SlicesPerVolume`.
- `data/tables/ADAS.csv` with `PTID`, `VISDATE`, and scored ADAS components.
The script filters the Mayo inventory to T1-weighted 3D acquisitions with `SlicesPerVolume >= --min-slices` (default 100). Missing or non-numeric slice counts fail the filter. In-plane dimensions (rows and columns) are not checked.
 
```
python MRI_ADAS-Cog11_Matching.py --min-slices 100 --n-images 10000 --output MRI_ADAS11_pairs.csv
```
 
The output contains `subject_id,image_id,image_date,score_date,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11`. The component columns map in order to `Q1SCORE,Q2SCORE,Q3SCORE,Q5SCORE,Q6SCORE,Q7SCORE,Q8SCORE,Q9SCORE,Q10SCORE,Q11SCORE,Q12SCORE`. Output `q4` is therefore source `Q5SCORE` (naming), not delayed recall. Delayed recall (`Q4SCORE`) and number cancellation (`Q13SCORE`) are excluded from the 11-item score.
 
Matching selects the closest complete assessment for the same participant within 90 days before or after the scan. Ambiguous matches are excluded. Sampling occurs after matching, with seed 42 by default. If fewer than 10,000 images qualify, the script exports all eligible pairs. Use `--n-images 0` for the full eligible list and `--help` for input paths, the date window, and sampling options.
 
Image IDs are written without the leading `I` (for example, `12345` rather than `I12345`). The `image_id` column can be converted to a comma-separated list for ADNI's image-ID filter; add the `I` prefix back if the filter requires it. Multiple images can share an assessment; keep all records from a participant in the same training, validation, or test split.
 
