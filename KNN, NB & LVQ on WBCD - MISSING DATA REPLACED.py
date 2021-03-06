#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from neupy import algorithms
from sklearn import metrics


# In[2]:


data_set = pd.read_csv("breast-cancer-wisconsin.data_MISSING_DATA_ROWS_REPLACED.csv")


# In[3]:


data_clean = data_set.dropna()


# In[4]:


X = data_clean.drop('target', axis=1)
y = data_clean['target']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)


# In[6]:


kNNModel = KNeighborsClassifier(n_neighbors=3)
kNNModel.fit(X_train, y_train)


# In[7]:


y_pred = kNNModel.predict(X_test)


# In[8]:


print(kNNModel.score(X_test, y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[9]:


nbModel = GaussianNB()
nbModel.fit(X_train, y_train)


# In[10]:


y_pred = nbModel.predict(X_test)


# In[11]:


print(nbModel.score(X_test, y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[12]:


lrModel = LogisticRegression(random_state=0, solver='lbfgs')
lrModel.fit(X_train, y_train)


# In[13]:


y_pred = lrModel.predict(X_test)


# In[14]:


print(lrModel.score(X_test, y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[16]:


data_set_lvq = pd.read_csv("breast-cancer-wisconsin.data_MISSING_DATA_ROWS_REPLACED_LVQ.csv")
data_clean_lvq = data_set_lvq.dropna()

_X = data_clean_lvq.drop('target', axis=1)
_y = data_clean_lvq['target']

_X_train, _X_test, _y_train, _y_test = train_test_split(_X, _y, test_size = 0.35)

lvqnet = algorithms.LVQ(n_inputs=9, n_classes=2)
lvqnet.train(_X_train, _y_train, epochs=100)
_y_pred = lvqnet.predict(_X_test)

print(confusion_matrix(_y_test,_y_pred))
print(metrics.accuracy_score(_y_test, _y_pred)*100.0)

