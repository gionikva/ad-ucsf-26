# ad-ucsf-26
Alzheimer's disease UCSF/Berkeley/BU research project 2026.
## Acquiring the data
Visit the [loni website](https://ida.loni.usc.edu) and apply for access to ADNI. Once access has been granted, navigate to the ARC Builder by clicking on the menu on the top right.

Download the MRIQC file and save it to `./data/tables/MRIQC.csv`. Run `python misc/gen_image_ids.py`, optionally with the `-n` parameter (I recommend starting with 5000) to generate a list of QC-passed images. Then, go back to the ADNI ARC builder, click on the `+` (create filter) button in the bottom pane and choose "select from image_ids you provide." Then, paste in the content of `images_sample.txt` and click apply.
Navigate to 
