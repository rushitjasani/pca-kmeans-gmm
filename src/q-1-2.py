#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


# In[4]:


Z_new.head()


# In[ ]:





# In[5]:


kmeans = KMeans(n_clusters=5, random_state=0).fit(Z_new)
cluster_scikit = kmeans.labels_
print cluster_scikit


# new_df = Z_new.iloc[:,:].values
# m = new_df.shape[0]
# n = new_df.shape[1]
# n_iter = 1
# K = 5

# Centroids=np.array([]).reshape(n,0)
# for i in range(K):
#     rand=rd.randint(0,m-1)
#     Centroids=np.c_[Centroids,new_df[rand]]
# print Centroids.shape
# #print Centroids
# print new_df.shape

# Output={}
# for i in range(n_iter):
#     EuclidianDistance=np.array([]).reshape(m,0)
#     for k in xrange(K):
#         tempDist=np.sum((new_df-Centroids[:,k])**2,axis=1)
#         EuclidianDistance=np.c_[EuclidianDistance,tempDist]
#     C=np.argmin(EuclidianDistance,axis=1)+1
# 
#     Y={}
#     for k in range(K):
#         Y[k+1]=np.array([]).reshape(n,0)
#     for i in range(m):
#         Y[C[i]]=np.c_[Y[C[i]],new_df[i]]
# 
#     for k in range(K):
#         Y[k+1]=Y[k+1].T
# 
#     for k in range(K):
#          Centroids[:,k]=np.mean(Y[k+1],axis=0)
#     Output=Y

# In[6]:


def euclidean_distance(x,y):
    return np.sum((x - y)**2)

def kmeans(K,df):

    d = df.shape[1] 
    n = df.shape[0]
    Max_Iterations = 100
    i = 0
    
    cluster = [0] * n
    prev_cluster = [-1] * n
    
    cluster_centers = [rd.choice(df) for i in xrange(K) ]    
    force_recalculation = False
    
    while (cluster != prev_cluster) or (i > Max_Iterations) or (force_recalculation) :
        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1
    
        for p in xrange(n):
            min_dist = float("inf")
            for c in xrange(K):
                dist = euclidean_distance(df[p],cluster_centers[c])
                if (dist < min_dist):
                    min_dist = dist  
                    cluster[p] = c
        
        for k in xrange(K):
            new_center = [0] * d
            members = 0
            for p in xrange(n):
                if (cluster[p] == k):
                    for j in xrange(d):
                        new_center[j] += df[p][j]
                    members += 1
            
            for j in xrange(d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members) 
                else: 
                    new_center = rd.choice(df)
                    force_recalculation = True                    
            
            cluster_centers[k] = new_center
    
        
    
#     print cluster_centers
#     print i
    return cluster


# In[7]:


cluster = kmeans(5, Z)


# In[8]:


Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )
Z_new = pd.concat([Z_new, Y], axis=1)

pred_Y = pd.DataFrame( cluster,columns=[ 'pred_Y' ] )
pred_Y_scikit = pd.DataFrame( cluster_scikit,columns=[ 'pred_Y' ] )

Z_mymodel = pd.concat([Z_new, pred_Y],axis=1)
Z_scikit  = pd.concat([Z_new, pred_Y_scikit],axis=1)

# Z_new.head(10)


# In[9]:


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


# In[10]:


purity_dict = purity(Z_mymodel,5)
print purity_dict


# In[11]:


purity_scikit = purity(Z_scikit,5)
print purity_scikit


# In[ ]:




