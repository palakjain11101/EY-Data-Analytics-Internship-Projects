#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries 
import numpy as np   
import pandas as pd  

dataset = pd.read_csv('C:/Users\Palak\Desktop\Restaurant_Reviews (2).tsv', delimiter = '\t') 

# library to clean data 
import re  
  
# Natural Language Tool Kit 
import nltk  
  
nltk.download('stopwords') 

from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer 
  
# Initialize empty array 
corpus = []  
  
# 1000 (reviews) rows to clean 
for i in range(0, 1000):  
    
    #clean text and convert all
    #words to lowercase
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  
    review = review.lower()  
      
    # split to array(default delimiter is " ") 
    review = review.split()  
      
    # creating PorterStemmer object to 
    # take main stem of each word 
    ps = PorterStemmer()  
      
    # loop for stemming each word 
    # in string array at ith row     
    review = [ps.stem(word) for word in review 
                if not word in set(stopwords.words('english'))]  
                  
    # rejoin all string array elements 
    # to create back into a string 
    review = ' '.join(review)   
      
    # append each string to create 
    # array of clean text  
    corpus.append(review) 


# In[ ]:


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
  
# To extract max 1500 features
cv = CountVectorizer(max_features = 1500)  
  
# X contains corpus
X = cv.fit_transform(corpus).toarray()  
  
# y contains answers if review 
# is positive or negative 
y = dataset.iloc[:, 1].values  


from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators = 1001, 
                            criterion = 'entropy') 
                              
model.fit(X_train, y_train) 

from sklearn.metrics import confusion_matrix 
  
cm = confusion_matrix(y_test, y_pred) 
  
cm 

