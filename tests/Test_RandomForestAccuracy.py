import numpy as np

from sklearn.model_selection import train_test_split
from tests.compare_accuracies import compare_accuracies

#Load data
data_path = "data/clean_data.csv"
accuracies_file_path  = "src/current_accuracies/random_forest/randomForest_accuracies.pkl"
actual_accuracies_file_path= "src/best_weights/randomForest_accuracies.pkl"

data = np.loadtxt(data_path, delimiter=",",dtype=float, skiprows=1)
col_names = np.genfromtxt(data_path , delimiter=',', names=True, dtype=float).dtype.names[1:31]

#Split inputs and labels
x_col_names = col_names[0:22]
y_col_names = col_names[22:30]

X = data[:, 1:23]
Y = data[:, 29:30]

#Split into test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=42)

#Define the maximum depth
DEPTH = 5

class TestRandomForestClass:

    def test_compare_accuracies(self):
        no_upper, lower_accuracies = compare_accuracies(accuracies_file_path, actual_accuracies_file_path)
        assert no_upper, "One or more accuracy is lower than the previous one. Key comparaison (actual vs new):\n{}".format("\n".join(lower_accuracies))

