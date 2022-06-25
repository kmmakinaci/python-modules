'''
Created on 11 Ara 2017

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


#Dataset Read
dataset = pd.read_csv('dataset.csv', delimiter = ';', low_memory=False)

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 8) 

#Decision Tree
print("#####     Decision Tree      #######")
clsf_dt = DecisionTreeClassifier(criterion = 'entropy') #max depth analizi yaplabilir ,   max_depth=3, random_state=0
clsf_dt.fit(X_train, Y_train)
Y_pred_dt = clsf_dt.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_dt))
#print ("predictions")
#print( Y_pred_dt)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_dt))
print ("classification report")
print(classification_report(Y_test, Y_pred_dt))
print("#####   ############   ######")

#Random Forest
print()
print("#####   Random Forest   ######")
clsf_rf = RandomForestClassifier(n_estimators=100, criterion = 'entropy' )
clsf_rf.fit(X_train, Y_train)
Y_pred_rf = clsf_rf.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_rf))
#print ("predictions")
#print( Y_pred_rf)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_rf))
print ("classification report")
print(classification_report(Y_test, Y_pred_rf))
print("#####   ############   ######")

##Naive Bayes
print("Naive Bayes")
clsf_nb = GaussianNB()
clsf_nb.fit(X_train, Y_train)
Y_pred_nb = clsf_nb.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_nb))
#print ("predictions")
#print( Y_pred_nb)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_nb))
print ("classification report")
print(classification_report(Y_test, Y_pred_nb))

#KNN
print()
print("####     KNN         ####")
clsf_knn = KNeighborsClassifier(n_neighbors = 5)
clsf_knn.fit(X_train, Y_train)
Y_pred_knn = clsf_knn.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_knn))
#print ("predictions")
#print( Y_pred_knn)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_knn))
print ("classification report")
print(classification_report(Y_test, Y_pred_knn))
print("#####   ############   ######")

##Logistic Regression
print("Logistic Regression")
clsf_lr = LogisticRegression()
clsf_lr.fit(X_train, Y_train)
Y_pred_lr = clsf_lr.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_lr))
#print ("predictions")
#print( Y_pred_knn)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_lr))
print ("classification report")
print(classification_report(Y_test, Y_pred_lr))


##Linear SVC
print("Linear SVC")
clsf_lsvc = SVC(kernel='linear')
clsf_lsvc.fit(X_train, Y_train)
Y_pred_lsvc = clsf_lsvc.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_lsvc))
#print ("predictions")
#print( Y_pred_lsvc)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_lsvc))
print ("classification report")
print(classification_report(Y_test, Y_pred_lsvc))

##Kernel SVC
print("Kernel SVC")
clsf_ksvc = SVC(kernel='rbf')
clsf_ksvc.fit(X_train, Y_train)
Y_pred_ksvc = clsf_ksvc.predict(X_test)
print ("confusion matrix")
print(confusion_matrix(Y_test, Y_pred_ksvc))
#print ("predictions")
#print( Y_pred_ksvc)
print ("accuracy score")
print(accuracy_score(Y_test, Y_pred_ksvc))
print ("classification report")
print(classification_report(Y_test, Y_pred_ksvc))


print("end")