#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import math
import warnings 
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


rollno = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
name = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", np.nan, np.nan, "k", "l", "m"]
marks = [40, 23, 50, 78, 48, 89, 90, 67, 84, 96, 76, np.nan, 97, np.nan, 65]
grade = ["F", "F", "P", "P", "P", "P", "P", "P", "P", "P", "P", "F", "P", np.nan, np.nan]
df = pd.DataFrame({"rollno" : rollno, "name" : name, "marks" : marks, "grade" : grade})
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


df.columns


# In[9]:


df.isna().sum()


# In[10]:


df.to_csv("academic_performance.csv")


# In[11]:


df.isna().sum()


# In[12]:


df["marks"] = df["marks"].fillna(df["marks"].mean())
df


# In[14]:


def fun1(value):
   return int(math.floor(value))
df["marks"] = df["marks"].apply(fun1)
df


# In[16]:


df = df[df['name'].notna()]
df


# In[18]:


for index, row in df.iterrows():
# print(row['marks'], row['grade'])
   if (row['marks'] > 40):
      df.loc[index, 'grade'] = 'P'
else:
      df.loc[index, 'grade'] = 'F'
df


# In[19]:


first_outlier = [16, 'n', 200, 'P']
second_outlier = [17, 'o', -100, 'F']
df.loc[15] = first_outlier
df.loc[16] = second_outlier
df


# In[20]:


sns.countplot(data=df, x=df['marks']);


# In[21]:


sns.boxplot(data=df, x='marks');


# In[22]:


from matplotlib.cbook import boxplot_stats
outliers = boxplot_stats(df['marks']).pop(0)['fliers']
outliers


# In[23]:


df


# In[25]:


df = df.drop([15,16], axis=0)
df


# In[ ]:




