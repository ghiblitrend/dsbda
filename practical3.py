#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print(df.head())
summary_stats = df.groupby('Pclass')[['Age', 'Fare']].describe()
print(summary_stats)


# In[3]:


import pandas as pd
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
df_iris = pd.read_csv(url, header=None)
df_iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print(df_iris.head())
setosa = df_iris[df_iris['species'] == 'Iris-setosa']
versicolor = df_iris[df_iris['species'] == 'Iris-versicolor']
virginica = df_iris[df_iris['species'] == 'Iris-virginica']
setosa_stats = setosa.describe()
versicolor_stats = versicolor.describe()
virginica_stats = virginica.describe()
print("Iris-setosa Statistics:")
print(setosa_stats)
print("\nIris-versicolor Statistics:")
print(versicolor_stats)
print("\nIris-virginica Statistics:")
print(virginica_stats)


# In[ ]:




