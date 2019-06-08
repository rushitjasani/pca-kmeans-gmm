#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import math


# In[2]:


df = pd.read_csv("../input_data/intrusion_detection/data.csv")
Y = df.xAttack
X = df.drop(['xAttack'],axis=1)
X = (X - X.mean())/X.std()


# In[3]:


cov_x = np.cov(X.T)

U,S,V = np.linalg.svd(cov_x)
S_sum = float(np.sum(S))

running_sum = 0
num_of_comp = 0
for i in xrange(len(S)):
    running_sum += S[i]
    if running_sum  / S_sum  >= 0.90:
        num_of_comp = i+1
        break

U_red = U[:,:num_of_comp]

Z = np.matmul(U_red.T, X.T)
Z = Z.T
Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )


# In[ ]:





# In[4]:


gmm = GaussianMixture(n_components=5, n_init=10 )


# In[5]:


gmm.fit(Z_new)


# In[6]:


# print gmm.means_
# print gmm.covariances_
class_var =  gmm.predict(Z_new)
print class_var.shape


# In[7]:


def purity(df,K):
    pur_dict = {}
    for i in xrange(K):
        sub_tab = df[ df['pred_Y'] == i ]
        name,count = np.unique(sub_tab['xAttack'],return_counts=True)
        mx_ind = np.argmax(count)
        print i , name[mx_ind]
        pur_dict[i] = count[mx_ind] / float(len(sub_tab))
        print name
        print count
        
    return pur_dict


# In[8]:


Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )
Z_new = pd.concat([Z_new, Y], axis=1)

pred_Y = pd.DataFrame( class_var,columns=[ 'pred_Y' ] )
Z_final = pd.concat([Z_new, pred_Y],axis=1)


# In[9]:


purity_dict = purity(Z_final,5)
print purity_dict


# In[ ]:




