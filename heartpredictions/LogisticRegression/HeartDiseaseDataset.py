import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from .create_dataloaders import create_dataloaders
from .compute_class_weights import compute_class_weights
from .compute_min_max import compute_min_max

"""
HeartDiseaseDataset Class

Dataset class for the heart diseases dataset. Inheritated from the torch Dataset class.

"""

class HeartDiseaseDataset(Dataset): 

    def __init__(self, path, any_disease=False, label_indexes=[30, 31], split_indexes=[1, 23, 23, 31]):
        """
        Parameters:
            path (str) : Dataset filepath.
            any_disease (boolean) : If True each row will take the max label of all labels, else dummies will be created from the label_indexes parameters.
            label_indexes (list) : List of begin and end position of label.
            split_indexes (list) : List of features and label position (size must be equal to 4. Example : [begin_pos_features, end_pos_features, begin_pos_label, end_pos_label].

        Atributes:
            data (array) : The dataset in a numpy array.
            x (tensor) : The dataset in a torch representation.
        """

        self.data = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(self.data[:, split_indexes[0]:split_indexes[1]])

        if any_disease:
            self.y = torch.from_numpy(np.amax(self.data[:, split_indexes[2]:split_indexes[3]], axis=1))
        else:
            self.y = torch.from_numpy(pd.get_dummies(self.data[:, label_indexes[0]:label_indexes[1]].flatten()).to_numpy(dtype=np.float32))

        self.minMax = torch.from_numpy(compute_min_max(path, split_indexes[0], split_indexes[1]))
        self.len = len(self.data)
        self.class_weights = compute_class_weights(self.y)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
            len (int)
        """
        return self.len

    def __getitem__(self, idx):
        """
        Normalize one item and return it.
        Normalization of each column between 0 and 1.

        Parameters:
            idx (index) : Index of the item to return.

        Returns:
            inputs (tensor), label (tensor)
        """
        inputs = torch.sub(self.x[idx], self.minMax[0])/ torch.sub(self.minMax[1],self.minMax[0])
        return inputs, self.y[idx]
