import numpy as np

#from heartpredictions.Tree.DecisionTree import DecisionTree
from heartpredictions.Tree.RandomForest import RandomForest
from sklearn.model_selection import train_test_split

data_path = "data/clean_data.csv"

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
    def test_init(self):
        model = RandomForest()
        assert model.num_trees == 5
        assert model.min_participant == 2
        assert model.max_depth == 2
        assert len(model.decision_trees) == 0

    def test_empty_num_trees(self, capsys):
        model = RandomForest(num_trees=0)
        captured = capsys.readouterr()
        assert captured.err == "Cannot create a forest without trees\n"

    def test_init_with_values(self):
        model = RandomForest(2, 3, 4)
        assert model.num_trees == 2
        assert model.min_participant == 3
        assert model.max_depth == 4  

    def test_compute_fit(self):
        model = RandomForest()
        model.fit(X, Y, x_col_names)
        assert len(model.decision_trees) == 5

    def test_compute_predict(self):
        model = RandomForest()
        model.fit(X, Y, x_col_names)
        predictions = model.predict(X)
        assert len(predictions) == 11627
        predictions = np.array(predictions)
        assert len(np.unique(predictions)) == 2
        assert 1 in np.unique(predictions)
        assert 0 in np.unique(predictions)