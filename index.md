## Welcome to the Heart Predictions Project

<!--You can use the [editor on GitHub](https://github.com/Dianevera/heart-prediction/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.-->

``` Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Tests](#Tests)
4. [State of the art](#State-of-the-art)
5. [Report of project](#Report-of-project)
6. [About the data analysis](#About-the-data-analysis)
7. [About the data cleaning](#About-the-data-cleaning)
8. [More about the heartpredictions module](#More-about-the-heartpredictions-module)
9. [Example of workflow](#Example-of-workflow)

# Introduction
Heart disease is one of the leading causes of death. Early detection is key to helping patients to mitigate the disease. For this project, we use synthetic clinical data made from the Farmingham study to train models to see which one works the best. This project was done in the context of our final semester AI project. The creators of this project are Ines Khemir, Diane Ngako and Jake Penney.

# Installation
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
The state of the art can be found in the file : https://www.dropbox.com/scl/fi/wt7suf37wg0n0s8wusnig/Study-of-posibilities.pptx?dl=0&rlkey=vni0fuk736trjp39ob8ebfqvz.

# Report of project
The project report can be found in the file : "TODO"

# Test integration
Thanks to the python application github's action we add a test integration action after each commit. 
  All test in the testsuite must be pass to merge a branch.
  To test localy you can just launch pytest in the root folder.
  
# About the data analysis

In the 

# About the data cleaning
#TODO 


# About the heartpredictions module

* # DecisionTree package
#TODO node decision tree

* # LogisticRegression package
#TODO Model trainer classweights dtaloader multiplelabelstrainer

* # MLP package
#TODO

# Examples
#TODO screen cast
