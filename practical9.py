#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='sex', y='age', hue='survived', data=titanic)

# Set plot labels and title
plt.title('Distribution of Age by Gender and Survival Status')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.grid(True)
plt.show()


# In[ ]:




