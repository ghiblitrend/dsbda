#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("train.csv")


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[9]:


df.isna().sum()


# In[10]:


df["Age"] = df["Age"].fillna(df["Age"].mean())
df.isna().sum()


# In[11]:


df["Name"]


# In[12]:


df["Sex"].value_counts()


# In[13]:


df["Ticket"].value_counts()


# In[14]:


df["Cabin"].value_counts()


# In[15]:


df["Embarked"].value_counts()


# In[19]:


def fun1(value):
    if (value == "male"):
        return 1
    else:
        return 0
def fun2(value):
    if (value == 'S'):
        return 0
    elif (value == 'C'):
        return 1
    elif (value == 'Q'):
        return 2
    else:
        return 0
df["Sex"] = df["Sex"].apply(fun1)
df["Embarked"] = df["Embarked"].apply(fun2)
df.isna().sum()


# In[20]:


df = df.drop("Cabin", axis=1)
df.shape


# In[23]:


plt.figure(figsize=(10,7))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True)
plt.show()


# In[24]:


df.info()


# In[25]:


sns.countplot(df["Survived"])
plt.show()


# In[27]:


sns.countplot(data=df, x="Pclass", hue="Survived")
plt.show()


# In[30]:


sns.countplot(data=df, x="Sex", hue="Survived", palette="Accent")
plt.show()


# In[32]:


sns.countplot(data=df, x="Embarked", hue="Survived")
plt.show()


# In[33]:


sns.histplot(df["Fare"])
plt.show()


# In[ ]:




