'''
Created on 9 Ara 2017

@author: mertmakinaci
'''
#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from pandas import plotting
#Dataset Read
dataset = pd.read_csv('dataset.csv', delimiter = ';', low_memory=False)

#Dataset Print
dataset.info()
print()
print("Dataset Shape")
print(dataset.shape)
dataset.head()
#

#Histograms
dataset.hist()
plt.show()
#

#Scatter plot matrix
#plotting.scatter_matrix(dataset)
#plt.show()
#

X = dataset.iloc[:, 1:18].values
Y = dataset.iloc[:, 18].values


print("Class Labels")
print(np.unique(Y)) #returns different class labels stored in dataset

#print(X[0])
#print(Y[0])