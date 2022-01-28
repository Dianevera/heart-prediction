import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from .compute_class_weights import compute_class_weights
from .compute_min_max import compute_min_max

class HeartDiseaseDataset(Dataset): 
    def __init__(self, path, any_disease=False, label_indexes=[30, 31], split_indexes=[1, 23, 23, 31]):
        """
            Computer the output tensor.

                    Parameters:
                            path (string): The dataset path
                            any_disease (bool): If we set up the network to predict all the diseases from one tensor instead of two
                            label_indexes ([int]): The indexes that the labels are on
                            split_indexes ([int]): The indexes we split the data and labels on

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
            Get the length of the datset.

                    Returns:
                            (int): The size of the datset
        """
        return self.len

    def __getitem__(self, idx):
        """
            Get the item in idx position in the datset.

                    Parameters:
                            idx (int): The position of the item

                    Returns:
                            inputs (Tensor): The data in idx position
                            self.y[idx] (Tensor): The label in idx position
        """
        inputs = torch.sub(self.x[idx], self.minMax[0])/ torch.sub(self.minMax[1],self.minMax[0])
        return inputs, self.y[idx]