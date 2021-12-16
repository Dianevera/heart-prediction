import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from heartpredictions.Tree.DecisionTree import DecisionTree
from heartpredictions.Tree.RandomForest import RandomForest

def prediction_analyse(model, X_test, Y_test, confusion_matrix_display=True, proportion_informations=True):
    """
    Predict and display tools to analyse the result.

    Args:
        X_test (array) : Inputs array.
        Y_test (array) : Labels array.
        confusion_matrix_display (bool) : If True display the confusion matrix
        proportion_informations (bool) : If True display proportion information (sensitivity, specificity, PPV, NPV)

    Returns:
        accuracy (float)
    """
    #Predict values for each individuals
    Y_pred = model.predict(X_test)

    #Compute accuracy score
    accuracy = accuracy_score(Y_test.flatten(), Y_pred)
    print("accuracy ==>", accuracy)

    #Confusion matrix
    if confusion_matrix_display:
        cm = confusion_matrix(Y_test.flatten(), Y_pred)

        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                    zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
        plt.show()

    #proportion informations
    if proportion_informations:
        TP = cm[1,1] # true positive 
        TN = cm[0,0] # true negatives
        FP = cm[0,1] # false positives
        FN = cm[1,0] # false negatives

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        PPV = TP / (TP + FP) if (TP + FP) != 0 else 'no positives values'
        NPV = TN / (TN + FN) if (TN + FN) != 0 else 'no negatives values'

        print(f'sensitivity : {sensitivity}, specificity : {specificity}, PPV : {PPV}, NPV : {NPV}')

    return accuracy


def evaluate(data_path, save_directory, model_name = "Decision Tree", num_trees = 0, min_participant=2, max_depth=2, pretty_print=False):
    """
    Lunch fit and predict for a given model

    Args:
        data_path(string): Data path
        save_directory(string): Directory where to save
        model_name(string): Model name "Decision Tree" or "Random forest"
        num_trees(int): Number of trees to buld in case of random forest
        min_participant(int): Minimum number of variables to compare individuals
        max_depth(int): Maximum depth
        pretty_print(boolean): Flag for printtint or not 
    """
    # Split dataset into train, validation and test
    data = np.loadtxt(data_path, delimiter=",",dtype=float, skiprows=1)
    col_names = np.genfromtxt(data_path , delimiter=',', names=True, dtype=float).dtype.names[1:31]
    x_col_names = col_names[0:22]
    y_col_names = col_names[22:30]

    X = data[:, 1:23]
    Y = data[:, 29:30]
    
    total_accuracy = 0
    models = []
    for i, column in enumerate(y_col_names):
        print("\x1b[6;30;41m\033[1m", column.center(30), "\x1b[0m\033[0m\n")
        i = i + 23
        Y = data[:, i:(i+1)]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4, random_state=42)
    
        print("\x1b[6;30;43m", f'{model_name} fit'.center(30), "\x1b[0m")
        
        if model_name == "Decision Tree":
            model = DecisionTree(min_participant=min_participant, max_depth=max_depth )
        else:
            model = RandomForest(num_trees=num_trees, min_participant=min_participant, max_depth=max_depth)

        model.fit(X_train, Y_train, x_col_names)

        if pretty_print and model_name == "Decision Tree":
            model.pretty_print()

        file_name = '_'.join([model_name.lower().replace(" ", "_"), column])
        model.save(os.path.join(save_directory, file_name))
        models.append(model)
        
        print("\n\x1b[6;30;43m", "Test".center(30),"\x1b[0m")
        
        total_accuracy += prediction_analyse(model, X_test, Y_test)
        print("\n\n\n")
    print("\x1b[6;30;42m", "Mean accuracy of all labels :".center(30), "\x1b[0m", total_accuracy / len(y_col_names))