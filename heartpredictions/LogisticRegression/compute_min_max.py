import numpy as np

def compute_min_max(data_path, first_col_index=1, last_col_index=23):
    """
    Get min and max values for each feature column.

    Parameters:
        data_path (str) : The dataset filepath.
        first_col_index (int) : Index of the first feature column.
        last_col_index (int) : Index of the last feature column.

    Returns:
        min_max (array)
    """
    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32, skiprows=1)[:, first_col_index:last_col_index]
    return np.stack((data.min(axis=0), data.max(axis=0)))
