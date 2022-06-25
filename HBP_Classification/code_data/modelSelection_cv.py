'''
Created on 14 Ara 2017

@author: mertmakinaci
'''

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#Dataset Read
dataset = pd.read_csv('dataset.csv', delimiter = ';', low_memory=False)

#Variables and variable settings
model_test = []
names = []
results = []
model_test.append(('DT',DecisionTreeClassifier()))
model_test.append(('RF',RandomForestClassifier()))
model_test.append(('NB',GaussianNB()))
model_test.append(('KNN',KNeighborsClassifier()))
model_test.append(('LR', LogisticRegression()))
model_test.append(('LinearSVC', SVC(kernel='linear')))
model_test.append(('KernelSVC', SVC(kernel='rbf')))

X = dataset.iloc[:, 1:18].values
Y = dataset.iloc[:, 18].values


#Data Preprocessing
LblEnc_Y = LabelEncoder()
Y = LblEnc_Y.fit_transform(Y)
LblEnc_X = LabelEncoder()
X[:, 0]= LblEnc_X.fit_transform(X[:, 0])
#print(X[0])
#print(Y[0])
StdScaler_X = StandardScaler()
X = StdScaler_X.fit_transform(X)
#print(X[0])

#Train and Test Data Creation
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size = 0.2, random_state = 8) 
#test and train parameters
fold = 10  #number of validation sub-dataset
seed = 7
scoring = 'accuracy' 


print ("evaluation of models with cross validation")
for name, model in model_test:
    kfold = model_selection.KFold(n_splits=fold, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Algorithm Comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
