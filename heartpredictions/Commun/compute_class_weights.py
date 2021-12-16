import numpy as np
import torch

def compute_class_weights(labels):
    """
        Calculate the class weights.

                Parameters:
                        labels (np array): The labels

                Returns:
                        ([float]): A list of all the class weights
    """
    unique, counts = np.unique(np.argmax(labels.numpy(), axis=1), return_counts=True)
    class_weights = torch.tensor([(class_counts / labels.shape[0]) for class_counts in counts],dtype=torch.float32)
    return class_weights
