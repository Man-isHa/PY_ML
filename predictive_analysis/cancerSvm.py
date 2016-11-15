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
predictors1=["radius","texture","perimeter","area","smoothness","compactness","concavity","concavePoints","symmetry","fractalDimensional"]
data.loc[data["diagnosis"] == "M", "diagnosis"] = 0
data.loc[data["diagnosis"] == "B", "diagnosis"] = 1
target=data["diagnosis"].astype(int,copy="True")
#print(data["diagnosis"].head(40))
#alg=svm.SVC(kernel="linear",C=.01)
#alg= svm.SVC(kernel='linear', C=10)
alg=svm.SVC(kernel='rbf', gamma=0.00001, C=100)
#alg = LogisticRegression(random_state=1)
kf=KFold(data.shape[0],n_folds=3,random_state=1)
'''
predictions=[]
for train,test in kf:
    
    train_predictors=(data[predictors].iloc[train,:])
    train_target=target.iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictions=alg.predict(data[predictors].iloc[test,:])
    predictions.append(test_predictions)

predictions = np.concatenate(predictions,axis=0)

predictions[predictions>0.5]=1
predictions[predictions<=0.5]=0

c=0
for i in range(len(predictions)):
#accuracy=sum(predictions[predictions==target])/len(predictions)
    if predictions[i]==target[i]:
        c=c+1
print(c/float(len(predictions)))
'''
scores=cross_validation.cross_val_score(alg,data[predictors1],target,cv=3)
print(scores.mean())

alg1=tree.DecisionTreeClassifier()
score1=cross_validation.cross_val_score(alg1,data[predictors1],target,cv=3)
print(score1.mean())


alg2=RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=8,min_samples_leaf=4)
score2=cross_validation.cross_val_score(alg2,data[predictors1],target,cv=3)
print(score2.mean())

