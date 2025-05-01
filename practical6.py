#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

iris = pd.read_csv(url, header=None, names=column_names)
iris.to_csv("iris.csv", index=False)
print("Dataset downloaded and saved as iris.csv")


# In[3]:


import os
print(os.path.exists("iris.csv"))  


# In[4]:


import pandas as pd
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

iris = pd.read_csv("iris.csv")
print(iris.head())


# In[5]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
iris['species'] = encoder.fit_transform(iris['species'])
print(iris['species'].unique()) 


# In[6]:


from sklearn.model_selection import train_test_split

X = iris.iloc[:, :-1]  
y = iris['species']   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# In[7]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Model trained and predictions made!")


# In[8]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# In[9]:


from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print(f'Accuracy: {accuracy:.2f}')
print(f'Error Rate: {error_rate:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




