#!/usr/bin/env python
# coding: utf-8

# ## In-Class Activity - Cyber Security Data Analysis 
# This notebook will guide you through the process of analyzing a cyber security dataset. Follow the TODO tasks to complete the assignment.
# 

# # Step 1: Importing the required libraries
# 
# TODO: Import the necessary libraries for data analysis and visualization.

# In[148]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
pd.options.display.max_rows = 999
warnings.filterwarnings("ignore")


# # Step 2: Loading the dataset
# 
# TODO: Load the given dataset.

# In[162]:


link = pd.read_csv("./Data/CySecData.csv")


# # Step 3: Display the first few rows of the dataset
# TODO: Import the necessary libraries for data analysis and visualization.

# In[163]:


link.head()


# # Step 4: Initial info on the dataset.
# 
# TODO: Provide a summary of the dataset.

# In[164]:


link.info()


# # Step 5: Creating dummy variables
# TODO: Create dummy variables for the categorical columns except for the label column "class".

# In[165]:


dfDummies = link[['protocol_type', 'service', 'flag']]


# # Step 6: Dropping the target column
# TODO: Drop the target column 'class' from the dataset.

# In[168]:


link_drop = link.drop('class', axis=1)
dfNumeric = link.select_dtypes(include=['number'])



# # Step 7: Importing the Standard Scaler
# TODO: Import the `StandardScaler` from `sklearn.preprocessing`.

# In[169]:


from sklearn.preprocessing import StandardScaler


# # Step 8: Scaling the dataset
# TODO: Scale the dataset using the `StandardScaler`.

# In[170]:


from sklearn.impute import SimpleImputer
import pandas as pd 

imputer = SimpleImputer(strategy='mean')
link_imputed = imputer.fit_transform(dfNumeric)

link_imputed = pd.DataFrame(link_imputed, columns=dfNumeric.columns, index=dfNumeric.index)

scale = StandardScaler()
dfNormalized = scale.fit_transform(link_imputed)


# # Step 9: Splitting the dataset
# TODO: Split the dataset into features (X) and target (y).

# In[172]:


X = dfNumeric
y = link['class'].copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Step 10: Importing the required libraries for the models
# TODO: Import the necessary libraries for model training and evaluation.

# In[173]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 


# # Step 11: Defining the models (Logistic Regression, Support Vector Machine, Random Forest)
# TODO: Define the models to be evaluated.

# In[176]:


from sklearn.metrics import accuracy_score, classification_report

models = []
models.append(('LR', LogisticRegression(max_iter=1000)))
models.append(('SVM', SVC(kernel='rbf', probability=True)))
models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=100, random_state=42)))


# # Step 12: Evaluating the models
# TODO: Evaluate the models using 10 fold cross-validation and display the mean and standard deviation of the accuracy.
# Hint: Use Kfold cross validation and a loop

# In[175]:


from sklearn.model_selection import KFold, cross_val_score
import numpy as np

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

    print(f"{name}: Mean Accuracy = {cv_results.mean():.4f}, Std = {cv_results.std():.4f}")


# # Step 13: Converting the notebook to a script
# TODO: Convert the notebook to a script using the `nbconvert` command.

# In[ ]:


get_ipython().system('jupyter nbconvert --to python cysec-data-analysis.ipynb')

