#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import pandas as pd


# In[8]:


california_housing = fetch_california_housing()


# In[9]:


df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['PRICE'] = california_housing.target


# In[10]:


print(df.head())


# In[11]:


correlation_matrix = df.corr()


# In[12]:


plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of California Housing Dataset")
plt.show()


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[14]:


X = df.drop('PRICE', axis=1)
y = df['PRICE']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


model = LinearRegression()


# In[19]:


model.fit(X_train_scaled, y_train)


# In[20]:


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


# In[21]:


from sklearn.metrics import mean_squared_error, r2_score


# In[22]:


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[23]:


r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")


# In[24]:


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()


# In[ ]:




