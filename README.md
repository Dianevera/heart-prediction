# heart-prediction
Heart attack risk prediction

#Installation
* ## Requirements installation
To install requirements:
```sh
python -m pip install -r requirements.txt
```

* ## Installation of the created module heartpredictions
To install all modules setup:
```sh
python -m pip install -e .
```

# Tests
* ## TestSuite usage 
To run the testsuite on linux: 
```sh
python -m pytest -v tests/Test*.py
```

Tests will fail as long as the weights giving a higher accuracy are not put in the best_weights folder

# State of the art
The state of the art can be found in the file : "TODO".

# Report of project
The project report can be found in the file : "TODO"

* ## Test integration
* Thanks to the python application github's action we add a test integration action after each commit. 
  All test in the testsuite must be pass to merge a branch.
  
# About the data analysis
#TODO 

# About the data cleaning
#TODO 


# About the heartpredictions module

* # DecisionTree package
#TODO node decision tree

* # LogisticRegression package
#TODO Model trainer classweights dtaloader multiplelabelstrainer

* # MLP package
#TODO

# Example of workflow
#TODO screen cast