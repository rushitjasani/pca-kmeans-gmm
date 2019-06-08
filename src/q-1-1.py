#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv("../input_data/intrusion_detection/data.csv")
Y = df.xAttack
X = df.drop(['xAttack'],axis=1)
X = (X - X.mean())/X.std()
X.head()


# In[3]:


pca = PCA(.90)
principalComponents = pca.fit(X)
X_red = pca.transform(X)
X_new = pd.DataFrame( X_red,columns=[ "pc"+str(i) for i in xrange(X_red.shape[1]) ] )


# In[4]:


finalDf = pd.concat([X_new, df[['xAttack']]], axis = 1)


# In[5]:


pca.explained_variance_ratio_


# In[ ]:





# In[6]:


cov_x = np.cov(X.T)
cov_x.shape


# In[7]:


U,S,V = np.linalg.svd(cov_x)
S_sum = float(np.sum(S))


# In[8]:


running_sum = 0
num_of_comp = 0
for i in xrange(len(S)):
    running_sum += S[i]
    if running_sum  / S_sum  >= 0.90:
        num_of_comp = i+1
        break
print num_of_comp


# In[9]:


print U.shape


# In[10]:


U_red = U[:,:num_of_comp]
print U_red.shape


# In[ ]:





# In[11]:


Z = np.matmul(U_red.T, X.T)
Z = Z.T
Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )


# In[ ]:





# In[12]:


X_new.head()


# In[13]:


Z_new.head()


# In[ ]:




