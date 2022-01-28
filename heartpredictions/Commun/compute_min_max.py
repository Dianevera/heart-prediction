import numpy as np

def compute_min_max(data_path, first_col_index=1, last_col_index=23):
    """
        Get the minimum and maximum value for each column in a range of a pd datframe.

                Parameters:
                        data_path (string): The data path
                        first_col_index (int): The start of the range
                        last_col_index (int): The end of the rnage

                Returns:
                        (np.stack): A stack of tuples of the min and max of each column
    """
    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32, skiprows=1)[:, first_col_index:last_col_index]
    return np.stack((data.min(axis=0), data.max(axis=0)))
