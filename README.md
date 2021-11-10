# heart-prediction
Heart attack risk prediction

# Requirements installation
To install requirements:
$ python -m pip install -r requirements.txt

# Installation of the created module heartpredictions
To install all modules setup:
$ python -m pip install -e .

# TestSuite usage
* Command to run the testsuite on linux: 
$ ./tests.sh
* Tests will fail as long as the weights giving a higher accuracy are not put in the best_weights folder

# Test integration
* Thanks to the python application github's action we add a tests integrations before a pull request merge and at each commit push

# About the heartpredictions module
## Logistic regression
The logistic regression from scratch with pytorch

Normalization of each value between 0 and 1 based on the minimun and the maximum of each column.
TODO
