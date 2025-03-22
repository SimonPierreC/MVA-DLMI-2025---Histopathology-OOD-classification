import h5py
import numpy as np


def open_file(path):
    with h5py.File(path, 'r') as hdf:
        idx_list = list(hdf.keys())
        labels = [np.array(hdf.get(idx).get("label")) for idx in idx_list]
        metadata = [np.array(hdf.get(idx).get("metadata")) for idx in idx_list]
        images = [np.array(hdf.get(idx).get("img")) for idx in idx_list]
    return idx_list, images, metadata, labels
