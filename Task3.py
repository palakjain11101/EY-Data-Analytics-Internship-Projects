#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

import numpy as np
from xlrd import open_workbook
from openpyxl.reader.excel import load_workbook

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


pd.options.display.precision = 2
pd.set_option('display.precision', 2)

file_name = "C:\\Users\\vyshnavi.garimella\\Desktop\\G_Tubes_Result_v0.xlsx"
workbook = open_workbook(file_name)

sheet_name = "Component Properties"

df = pd.read_excel(file_name, sheet_name)

df = df[["Technology", "Property","Anomaly"]]

msk = np.random.rand(len(df)) < 0.8

train = df[msk]

test = df[~msk]

print(len(train))
print(len(test))

combine = [train, test]
technology_mapping = {"Tube - Laminate Threaded": 1, "Tube - Laminate Snap-On": 2, "Tube - Bayonet": 3, 
                      "Tube - Aluminum": 4, "Tube - Extruded Snap-On": 5, "Tube - Extruded Threaded": 6}

property_mapping = {"*Bore Diameter": 1, "*Burst Strength": 2, "*Lamination Strength": 3, "*Overlap Width": 4,
                    "*Removal Force": 5, "*Removal Torque": 6, "*Side Seam Compression": 7}

anomaly_mapping = {"Yes": 1, "No": 0}



for dataset in combine:
    dataset['Technology']= dataset['Technology'].map(technology_mapping)

    dataset['Property']= dataset['Property'].map(property_mapping)
    

    dataset['Anomaly']= dataset['Anomaly'].map(anomaly_mapping)


train.reset_index(inplace = True) 
train.head()
    
train = train.drop(["index"], axis  = 1)

test.reset_index(inplace = True) 
test.head()

print("Train data ....")
print(train.head())
print("Test data ....")
print(test.head())


print(train.shape)
print(test.shape)
print(df.shape)


train = train.dropna()

test = test.dropna()


X_train = train.drop("Anomaly", axis=1)
Y_train = train["Anomaly"]
X_test  = test.drop(["index","Anomaly"], axis=1)

X_train.shape, Y_train.shape, X_test.shape


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

coeff_df = pd.DataFrame(train.columns.delete(2))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian

models = pd.DataFrame({'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)

