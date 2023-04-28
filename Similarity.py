#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_excel(r'C:\Users\Rif010\Desktop\Clustering\Cluster.xlsx', sheet_name="People Records")
df.head()


# In[3]:


df.head(100)
df.shape


# In[4]:


Feature = pd.get_dummies(df['SuperType'])

X=Feature[0:50]
Y=Feature[51:100]

X.shape


# In[5]:


X.head()


# In[6]:


X=X.sum(axis=0)

Y=Y.sum(axis=0)

X


# In[7]:


x=X.to_numpy()
x
y=Y.to_numpy()
x=x.reshape(1,-1)
y=y.reshape(1,-1)
x,y


# # Cosine Similarity

# In[8]:


from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(x,y))


# # Pearson Correlation

# In[9]:


from scipy.stats import pearsonr
corr,_=pearsonr(X,Y)
print(corr)


# In[ ]:





# In[ ]:




