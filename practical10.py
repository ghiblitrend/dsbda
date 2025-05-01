#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Iris dataset
iris = sns.load_dataset('iris')

# Quick look
print(iris.head())


# In[2]:


# Histograms for all numeric features
iris.hist(edgecolor='black', figsize=(10, 8))
plt.suptitle('Feature Distributions in Iris Dataset')
plt.show()


# In[3]:


# Boxplots for all numeric features
plt.figure(figsize=(10, 6))
for i, column in enumerate(iris.columns[:-1]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=iris[column])
    plt.title(f'Boxplot of {column}')
plt.tight_layout()
plt.show()


# In[ ]:




