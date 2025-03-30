import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import random
import os

class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        
        with h5py.File(self.dataset_path, 'r') as hdf:        
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = torch.tensor(hdf.get(img_id).get('img'))
            label = np.array(hdf.get(img_id).get('label')) if self.mode == 'train' else None
        if self.preprocessing is not None:
            img = self.preprocessing(img)
        return img.float(), label

class PrecomputedDataset(Dataset):
    def __init__(self, features, labels):
        super(PrecomputedDataset, self).__init__()
        self.features = features
        self.labels = labels.unsqueeze(-1)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx].float()

class EmbeddedDomainAwareDataset(Dataset):
    def __init__(self, dataset_path, embedded_dataset_path, preprocessing, mode):
        super(EmbeddedDomainAwareDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode
        self.embedded_dataset_path = embedded_dataset_path
        self.image_ids = []
        self.domains = []
        
        # Load the dataset and extract image IDs and domain information
        for imgs_h5 in os.listdir(self.dataset_path):
            file_number = imgs_h5.split('.')[0].split('_')[1]
            with h5py.File(self.dataset_path + '/' + imgs_h5, 'r') as hdf:
                position_in_file = 0
                for img_id in list(hdf.keys()):
                    self.image_ids.append((img_id, file_number, position_in_file))
    
                    # Extract domain information
                    metadata = np.array(hdf.get(img_id).get('metadata'))
                    self.domains.append(int(metadata[0]))
                    position_in_file += 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id, file_number, position_in_file = self.image_ids[idx]
        embedded_path = self.embedded_dataset_path + '/' + self.mode + "_" + file_number + '.pth'
        imgs = torch.load(embedded_path, weights_only=False)
        img, label=  imgs[position_in_file]
        domain = self.domains[idx]
        if self.preprocessing is not None:
            img = self.preprocessing(img)
        # Convert domain to a tenso
        domain = torch.tensor(domain).unsqueeze(0)
        return img.float(), label, domain
