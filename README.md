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

To preoprocess the data, run `python preprocess.py -d 192`. In case the script crashes or is interrupted, you may resume preprocessing with `python preprocess.py -d 192 -r`, though do note this functionality has not been extensively tested so I would use it with caution. In general, you may `python preprocess.py --help` for all available options.

## Segmentation pretraining
To train the (LightMedSeg) backbone on the BM/WM/CSF segmentation pretraining task, run 

```python train_seg.py -o weights/<placeholder> --batch-size <BATCH_SIZE> -e 100 -s medium -a --deep-supervision```.


By default, this will *not* downsample the input `192^3` volumes to avoid destroying fine detail, which might result in out of memory errors and very slow training. If this does occur, simply add the `-d` or `--downsample` parameter to enable downsampling in the model. 
