import os
from os import scandir


def get_dataset_filepaths(path, range=None):
    metadata = []
    features = []
    labels = []

    root = path
    dirs = [f for f in scandir(root)]

    for dir in dirs:
        path = dir.path

        metadata.append(os.path.join(path, "meta.csv"))
        features.append(os.path.join(path, "img.nii.gz"))
        labels.append(os.path.join(path, "mask.nii.gz"))

    if range is not None:
        metadata = metadata[range[0]:range[1]]
        features = features[range[0]:range[1]]
        labels = labels[range[0]:range[1]]
        
    return metadata, features, labels