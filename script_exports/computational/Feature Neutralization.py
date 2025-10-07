#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[4]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# Load ratios - takes around 5 mins to load data 
df_accounting = sov.data("accounting/weekly", start_date="2024-01-26")

# Choose long enough history for the model to train
df_mega = df_accounting.select_stocks("mega")

# # Feature Neutralization
# 
# All these methods return the same number of columns as the input DataFrame. They transform the data while maintaining the original dimensionality, which is crucial for many financial applications where each feature represents a specific economic or financial metric. 
# 
# 1. Orthogonalization might be preferred when you want to remove correlations but keep the overall structure of the data.
# 1. Neutralization might be used when you want to focus on the unique aspects of each feature, removing common market factors.

# ### Orthogonalization
# 
# Orthogonalization transforms a set of features into a new set of uncorrelated (perpendicular) features while preserving the original information content.

# In[5]:


# Gram-Schmidt method
df_mega.orthogonalize_features(method='gram_schmidt')

# In[6]:


# QR method
df_mega.orthogonalize_features(method='qr')

# ### Neutralization
# Neutralization reduces the influence of common factors across features, typically by removing one or more principal components, leaving only the unique aspects of each feature.

# In[7]:


# PCA method
df_mega.neutralize_features(method='pca')

# In[8]:


# SVD method
df_mega.neutralize_features(method='svd')

# In[9]:


# Iterative Regression method (very slow method, run with small datasets)
# df_mega.neutralize_features(method='iterative_regression')
