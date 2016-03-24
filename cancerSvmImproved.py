# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:55:29 2016

@author: Manisha
"""

import numpy as np
import pandas
from pandas import DataFrame
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
data= pandas.read_csv("cancer.csv")
data.columns=["idNumber","diagnosis","radius1","radius2","radius3","texture1","texture2","texture3","perimeter1","perimeter2","perimeter3","area1","area2","area3","smoothness1","smoothness2","smoothness3","compactness1","compactness2","compactness3","concavity1","concavity2","concavity3","concavePoints1","concavePoints2","concavePoints3","symmetry1","symmetry2","symmetry3","fractalDimensional1","fractalDimensional2","fractalDimensional3"]
data["radius"]=(data["radius1"]+data["radius2"]+data["radius3"])/3
data["texture"]=(data["texture1"]+data["texture2"]+data["texture3"])/3
data["perimeter"]=(data["perimeter1"]+data["perimeter2"]+data["perimeter3"])/3
data["area"]=(data["area1"]+data["area2"]+data["area3"])/3
data["smoothness"]=(data["smoothness1"]+data["smoothness2"]+data["smoothness3"])/3
data["compactness"]=(data["compactness1"]+data["compactness2"]+data["compactness3"])/3
data["concavePoints"]=(data["concavePoints1"]+data["concavePoints2"]+data["concavePoints3"])/3
data["concavity"]=(data["concavity1"]+data["concavity2"]+data["concavity3"])/3
data["symmetry"]=(data["symmetry1"]+data["symmetry2"]+data["symmetry3"])/3
data["fractalDimensional"]=(data["fractalDimensional1"]+data["fractalDimensional2"]+data["fractalDimensional3"])/3
#print(data["diagnosis"].head(5))
predictors=["radius1","radius2","radius3","texture1","texture2","texture3","perimeter1","perimeter2","perimeter3","area1","area2","area3","smoothness1","smoothness2","smoothness3","compactness1","compactness2","compactness3","concavity1","concavity2","concavity3","concavePoints1","concavePoints2","concavePoints3","symmetry1","symmetry2","symmetry3","fractalDimensional1","fractalDimensional2","fractalDimensional3"]
predictors1=["radius","texture","perimeter","concavity","concavePoints","fractalDimensional"]
data.loc[data["diagnosis"] == "M", "diagnosis"] = 0
data.loc[data["diagnosis"] == "B", "diagnosis"] = 1
target=data["diagnosis"].astype(int,copy="True")


algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), predictors]
]

full_predictions = []
for alg, predictors in algorithms:
    # Fit the algorithm using the full training data.
    alg.fit(data[predictors], target)
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(data[predictors].astype(float))[:,1]
    full_predictions.append(predictions)
    

# The gradient boosting classifier generates better predictions, so we weight it higher.
predictions = (full_predictions[0] *3 + full_predictions[1]) / 4
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
predictions = predictions.astype(int)
c=0
for i in range(len(predictions)):
#accuracy=sum(predictions[predictions==target])/len(predictions)
    if predictions[i]==target[i]:
        c=c+1
print(c/float(len(predictions)))










