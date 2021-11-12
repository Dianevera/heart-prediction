import ast
import numpy as np
import os
import sys

from heartpredictions.Tree.DecisionTree import DecisionTree

class RandomForest():
    def __init__(self, num_trees = 5, min_participant=2, max_depth=2):
        """
        Attributes:
            num_trees(int): number of trees to build
            min_participant(int): Minimum number of variables to compare individuals.
            max_depth(int): Maximun depth of a tree
        """
        if (num_trees == 0):
            print("Cannot create a forest without trees", file=sys.stderr)
        self.num_trees = num_trees
        self.min_participant = min_participant
        self.max_depth = max_depth
        self.decision_trees = []

    def random_choice(self, X, y):
        """
        Generate a random dataset from an other

        Args:
            X(array): The dataset
            y(array): The labels
        Return:
            X(array): The random selected dataset
            y(array): The random selected labels
        """
        n_rows, _ = X.shape
        choices = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[choices], y[choices]
        
    def fit(self, X, y, x_col_names):
        """
        Train the model

        Args:
            X(array): Input dataset
            y(array): Labels
            x_col_names(list): List of column name
        """
        
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        built_tree = 0
        # Build each tree of the forest
        while built_tree < self.num_trees:
            # Avoid exception if random dataset is not correct
            try:
                decision_tree = DecisionTree(
                    min_participant=self.min_participant,
                    max_depth=self.max_depth
                )
                X_train, y_train = self.random_choice(X, y)
                decision_tree.fit(X_train, y_train, x_col_names)
                self.decision_trees.append(decision_tree)
                built_tree += 1
            except Exception as e:
                continue
    
    def predict(self, X):
        """
        Make prediction

        Args:
            X(array): Input dataset
        Returns:
            predictions(list): List of predictions
        """
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
   
        # Reshape so we can find the most common value
        y = np.swapaxes(a=y, axis1=0, axis2=1)

        # Use mean for the final prediction
        predictions = []
        for preds in y:
            mean = np.mean(preds)
            predictions.append(int(round(mean)))
        return predictions

    def save(self, path):
        """
        Save the random forest in a file

        Args:
            path(string): File path
        """
        file = open(path, "w+")
        file.write(str([self.num_trees, self.min_participant, self.max_depth]))
        file.close()
        for tree in self.decision_trees:
            tree.save(path, random_forest_call = True)

    def load_from_file(self, path):
        """
        Load a forest from a given path

        Args:
            path(string): File path 
        """
        if not os.path.isfile(path):
            print("File does not exist :", path, file=sys.stderr)
            
        with open(path, "r") as file:
            forest_attributes = ast.literal_eval(file.readline())
            self.num_trees = forest_attributes[0]
            self.min_participant = forest_attributes[1]
            self.max_depth = forest_attributes[2]
            
            for i in range(self.num_trees):
                tree_list = ast.literal_eval(file.readline())
                tree = DecisionTree()
                tree.x_col_names = tree_list[0]
                tree.min_participant = tree_list[1]
                tree.max_depth = tree_list[2]
                tree_list = tree_list[3]
                tree.root = tree.build_tree_from_file(tree_list)
                self.decision_trees.append(tree)