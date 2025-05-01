#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


np.random.seed(42)


# In[3]:


num_samples = 1000
ages = np.random.randint(18, 60, size=num_samples)
salaries = np.random.uniform(30000, 120000, size=num_samples)
purchases = np.random.randint(0, 2, size=num_samples) 


# In[5]:


df = pd.DataFrame({
    'Age': ages,
    'EstimatedSalary': salaries,
    'Purchased': purchases
})
print(df.head())


# In[6]:


X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[9]:


model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)


# In[10]:


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)

TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]


# In[11]:


accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy: {accuracy}")
print(f"Error Rate: {error_rate}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of Age vs. Estimated Salary, colored by the Purchased target
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='EstimatedSalary', hue='Purchased', palette='coolwarm', s=100, alpha=0.7)
plt.title("Age vs Estimated Salary (Colored by Purchase)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend(title='Purchased', loc='upper right')
plt.show()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=["Not Purchased", "Purchased"], yticklabels=["Not Purchased", "Purchased"])
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:



