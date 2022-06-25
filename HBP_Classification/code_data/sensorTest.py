'''
Created on 21 Ara 2017

@author: mertmakinaci
'''

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
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Dataset Read
dataset = pd.read_csv('dataset_onlySens.csv', delimiter = ';', low_memory=False)

#dataset.drop(dataset.columns[[9, 10, 11]], axis=1)  # df.columns is zero-based pd.Index 

X = dataset.iloc[:, 1:13].values
Y = dataset.iloc[:, 13].values

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

#test = SelectKBest(score_func=chi2, k=9)
#fit = test.fit(X, Y)
# summarize scores
#np.set_printoptions(precision=3)
#print(fit.scores_)
#X_f = fit.transform(X)

#Train and Test Data Creation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8) 

model_accuracies = { 'DT':1, 'RF':1, 'KNN':1 ,'KernelSVC':1 }

#Decision Tree
clsf_dt = DecisionTreeClassifier(criterion = 'entropy') #max depth analizi yaplabilir ,   max_depth=3, random_state=0
clsf_dt.fit(X_train, Y_train)
Y_pred_dt = clsf_dt.predict(X_test)

#Random Forest
clsf_rf = RandomForestClassifier(n_estimators=100, criterion = 'entropy' )
clsf_rf.fit(X_train, Y_train)
Y_pred_rf = clsf_rf.predict(X_test)

#KNN
clsf_knn = KNeighborsClassifier(n_neighbors = 5)
clsf_knn.fit(X_train, Y_train)
Y_pred_knn = clsf_knn.predict(X_test)

##Kernel SVC
clsf_ksvc = SVC(kernel='rbf')
clsf_ksvc.fit(X_train, Y_train)
Y_pred_ksvc = clsf_ksvc.predict(X_test)


#accuracies
print("Model Accuracies")
model_accuracies['DT'] = accuracy_score(Y_test, Y_pred_dt)
model_accuracies['RF'] = accuracy_score(Y_test, Y_pred_rf)
model_accuracies['KNN'] = accuracy_score(Y_test, Y_pred_knn)
model_accuracies['KernelSVC'] =accuracy_score(Y_test, Y_pred_ksvc)

print(model_accuracies)

print("end")