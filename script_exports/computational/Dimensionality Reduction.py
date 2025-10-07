#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[6]:


import polars as pl

# In[7]:


pl.__version__

# In[1]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# Load ratios - takes around 5 mins to load data 
df_mega = sov.data("accounting/weekly", start_date="2021-01-26").select_stocks("mega")
df_mega.shape

# **PCA** (Principal Component Analysis): Reduces the data to `n_components` dimensions by projecting it onto the top `n_components` directions that maximize variance.

# In[2]:


df_mega.reduce_dimensions(method="pca", n_components=10)

# **Gaussian Random Projection**: Reduces the data to `n_components` dimensions by projecting it onto a randomly generated Gaussian matrix while preserving the pairwise distances between points.

# In[3]:


df_mega.reduce_dimensions(method="gaussian_random_projection", n_components=10)

# **Factor Analysis** reduces dataset dimensionality by representing correlated variables with fewer `n_components` unobserved variables, known as factors.

# In[5]:


df_mega.reduce_dimensions(method="factor_analysis", verbose=True, n_components=10)
