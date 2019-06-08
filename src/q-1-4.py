#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

df = pd.read_csv("../input_data/intrusion_detection/data.csv")
Y = df.xAttack
X = df.drop(['xAttack'],axis=1)
X = (X - X.mean())/X.std()
X.head()

pca = PCA(.90)
principalComponents = pca.fit(X)
X_red = pca.transform(X)
X_new = pd.DataFrame( X_red,columns=[ "pc"+str(i) for i in xrange(X_red.shape[1]) ] )

finalDf = pd.concat([X_new, df[['xAttack']]], axis = 1)

pca.explained_variance_ratio_

cov_x = np.cov(X.T)
cov_x.shape

U,S,V = np.linalg.svd(cov_x)
S_sum = float(np.sum(S))

running_sum = 0
num_of_comp = 0
for i in xrange(len(S)):
    running_sum += S[i]
    if running_sum  / S_sum  >= 0.90:
        num_of_comp = i+1
        break

# print num_of_comp

# print U.shape

U_red = U[:,:num_of_comp]
# print U_red.shape

Z = np.matmul(U_red.T, X.T)
Z = Z.T
Z_new = pd.DataFrame( Z,columns=[ "pc"+str(i) for i in xrange(Z.shape[1]) ] )



# In[2]:


agm = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single').fit(Z_new)
cluster_agm = agm.labels_


# In[3]:


kmeans = KMeans(n_clusters=5, random_state=0).fit(Z_new)
cluster_k_means = kmeans.labels_


# In[4]:


gmm = GaussianMixture(n_components=5, n_init=10 ).fit(Z_new)
cluster_gmm =  gmm.predict(Z_new)


# In[5]:


def purity(df,pred_Y,K):
    pred_Y = pd.DataFrame( pred_Y,columns=[ 'pred_Y' ] )
    df = pd.concat([df, pred_Y],axis=1)
    pur_dict = {}
    for i in xrange(K):
        sub_tab = df[ df['pred_Y'] == i ]
        name,count = np.unique(sub_tab['xAttack'],return_counts=True)
        mx_ind = np.argmax(count)
#         print i , name[mx_ind]
        pur_dict[i] = count[mx_ind] / float(len(sub_tab))
#         print name
#         print count
        
    return pur_dict
  


# In[6]:


Z_new = pd.concat([Z_new, Y], axis=1)

purity_dict_agm = purity(Z_new, cluster_agm ,5)
purity_dict_k_means = purity(Z_new, cluster_k_means ,5)
purity_dict_gmm = purity(Z_new, cluster_gmm ,5)
# print purity_dict


# In[7]:


def plot_func(purity_dict):
    val = []
    label = []
    for k,v in purity_dict.items():
      val.append(v)
      label.append(k)

    print val,
    print label

    plt.pie( val, labels = label)
    plt.show()


# In[8]:


plot_func(purity_dict_agm)
plot_func(purity_dict_gmm)
plot_func(purity_dict_k_means)


# In[ ]:




