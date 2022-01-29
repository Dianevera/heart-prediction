# Welcome to the Heart Predictions Project

<!--You can use the [editor on GitHub](https://github.com/Dianevera/heart-prediction/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.-->

### Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Tests](#Tests)
4. [Data](#Data)
5. [State of the art](#State-of-the-art)
6. [Paper of the project](#Paper-of-the-project)
7. [About the data analysis](#About-the-data-analysis)
8. [About the data cleaning](#About-the-data-cleaning)
9. [More about the heartpredictions module](#More-about-the-heartpredictions-module)
10. [Example of workflow](#Example-of-workflow)


# Introduction
Heart disease is one of the leading causes of death. Early detection is key to helping patients to mitigate the disease. For this project, we use synthetic clinical data made from the Farmingham study to train models to see which one works the best. This project was done in the context of our final semester AI project. The creators of this project are Ines Khemir, Diane Ngako and Jake Penney.

# Installation
This project is compatible with python 3.9
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

# Data  
The data is a synthetic verion of the Framingham datset. It is only for educational use only. To download it follow this link <a href="https://www.dropbox.com/s/dyazza8xhfjjcx3/frmgham2.csv?dl=0" target="_top">link</a>
Download the data and put the csv file in data/.

# State of the art
The state of the art can be found at this <a href="https://www.dropbox.com/scl/fi/wt7suf37wg0n0s8wusnig/Study-of-posibilities.pptx?dl=0&rlkey=vni0fuk736trjp39ob8ebfqvz" target="_top">link</a>

# Paper of the project
The project report can be found in <a href="www.dropbox.com/s/52id1lqvinhp5bt/Heart_Predictions.pdf?dl=0" target="_top">PDF</a> and <a href="https://www.dropbox.com/s/4cerqlftwekyl2o/Heart%20Predictions.zip?dl=0" target="_top">LaTex</a>

# Test integration
Thanks to the python application github's action we add a test integration action after each commit. 
  All test in the testsuite must be pass to merge a branch.
  
# About the data analysis
In the data analysis we analyse the data in order to understand the problem and help our intuiting for the algorithm part of the project.
The data analysis is found in the notebook located at : src/Data_analysis.ipynb

# About the data cleaning
For the datacleaning we remove columns with insufficient data and fill the rest in using a KNN.
The data cleaning can ba found at : src/data_cleaning.py
Eventhough we discontinued the LSTM implementation you can checkout our datacleaner that needs to be ran before the using the LSTM at : src/data_cleaning_lstm.py

To use the datacleaner use this command at the root of the project:
```sh
python src/data_cleaning.py
```

# About the heartpredictions module

* # Tree package
	In this package we have all our function that are unique to the decision tree and random forest models. This package is located at : /heartprediction/Tree

* # LogisticRegression package
	In this package we have all our function that are unique to the logistic regression model. This package is located at : /heartprediction/LogisticRegression

* # LSTM package
	In this package we have all our function that are unique to the LSTM model. Eventhough this is discontinued you are free to check out the implementation. This package is located at : /heartprediction/LogisticRegression

* # MLP package
	In this package we have all our function that are unique to the MLP model. This package is located at : /heartprediction/LogisticRegression

# Examples
<a href="https://youtu.be/n01OsuZs47E" target="_top">Presentation video</a>
