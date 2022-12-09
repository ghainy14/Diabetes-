#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:



df= pd.read_csv("health_data.csv")


# In[5]:


df.info()


# In[6]:


df


# In[7]:


df.isnull().sum()


# In[8]:


df.isna().sum()


# In[9]:


df.head


# In[10]:


df.head(5)


# In[11]:


df.describe()


# In[12]:


df.drop('Hypertension',axis=1, inplace=True)


# In[13]:


df.drop('Stroke',axis=1, inplace=True)


# In[14]:


df.head(5)


# In[15]:


df.info()


# In[16]:


x=df["BMI"]
y=df["Diabetes"]
plt.plot(x,y)
plt.title('Graph of Diabetes', fontdict={'fontsize':30, 'fontname':'arial', 'color' :'green'})
xlabel="Age"
ylabel="Diabetes"
plt.show()


# In[17]:


X=df.drop('Diabetes',axis=1)
y=df['Diabetes']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)


# In[20]:


X_train.shape,y_train.shape


# In[21]:


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X_train,y_train)
prediction=linear.predict(X_test)


# In[22]:


prediction


# In[ ]:




