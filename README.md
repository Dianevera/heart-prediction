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
$ python -m pytest -v tests/Test*.py
* Tests will fail as long as the weights giving a higher accuracy are not put in the best_weights folder

# Test integration
* Thanks to the python application github's action we add a tests integrations before a pull request merge and at each commit push
