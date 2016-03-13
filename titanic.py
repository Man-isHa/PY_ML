# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:01:38 2016

@author: Manisha
"""
import numpy as np
import pandas
from pandas import DataFrame
import re
import operator
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble
titanic = pandas.read_csv("train.csv")

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#print(titanic.describe())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
#print(titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=150, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
# Generating a familysize column
titanic["FamilySize"]=titanic["SibSp"]+titanic["Parch"]
titanic["NameLength"]=titanic["Name"].apply(lambda x: len(x))

def get_title(name):
    title_search=re.search('([A-Za-z]+)\.',name)
    if title_search:
        return title_search.group(1)
    return ""
    
titles=titanic["Name"].apply(get_title)
print(pandas.value_counts(titles))

title_mapping={"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles==k]=v
    
print(pandas.value_counts(titles))

titanic["Title"]=titles    
family_id_mapping={}

def get_family_id(row):
    last_name=row["Name"].split(",")[0]
    family_id="{0}{1}".format(last_name,row["FamilySize"])
    if family_id not in family_id_mapping:
        if len(family_id_mapping)==0:
            current_id=1
        else:
            current_id=(max(family_id_mapping.items(),key=operator.itemgetter(1))[1]+1)
        family_id_mapping[family_id]=current_id
    return family_id_mapping[family_id]
    
family_ids=titanic.apply(get_family_id,axis=1)
family_ids[titanic["FamilySize"]<3]=-1
print(pandas.value_counts(family_ids))
titanic["FamilyId"]=family_ids

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

selector=SelectKBest(f_classif,k=5)
selector.fit(titanic[predictors],titanic["Survived"])
scores=-np.log10(selector.pvalues_)
plt.bar(range(len(predictors)),scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

predictors = ["Pclass", "Sex", "Fare", "Title"]
alg=RandomForestClassifier(random_state=1,n_estimators=150,min_samples_split=8,min_samples_leaf=4)
scores=cross_validation.cross_val_score(alg,titanic[predictors],titanic["Survived"],cv=3)
print(scores.mean())



