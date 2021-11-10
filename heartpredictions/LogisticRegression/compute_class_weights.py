import numpy as np
import torch

def compute_class_weights(labels):
    """
    Compute the weights of each class in the given array.

    Parameters:
        labels (array) : The labels array.

    Returns:
        class_weights (Tensor)
    """
    unique, counts = np.unique(np.argmax(labels.numpy(), axis=1), return_counts=True)
    class_weights = torch.tensor([(class_counts / labels.shape[0]) for class_counts in counts],dtype=torch.float32)
    return class_weights
