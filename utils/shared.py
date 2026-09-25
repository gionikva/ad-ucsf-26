import os
from pathlib import Path
from os import scandir


def get_image_dirs(root: str, range=None):
    dirs = []

    for subject in scandir(root):
        for image in scandir(subject.path):
            dirs.append(image.path)
        
    return dirs