#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('test.csv')
df.head()


# In[3]:


missing_values = df.isnull().sum()
print(missing_values)


# In[4]:


statistics = df.describe()
print(statistics)


# In[5]:


print(df.dtypes)


# In[ ]:




