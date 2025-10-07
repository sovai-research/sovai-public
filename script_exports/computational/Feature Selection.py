#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# Load ratios - takes around 5 mins to load data 
df_accounting = sov.data("accounting/weekly", start_date="2024-01-26")

# Choose long enough history for the model to train
df_mega = df_accounting.select_stocks("mega")

# ### Feature Selection

# **Random Projection:**
# The feature importance reflects how much each feature contributes to the variance in the randomly projected space.
# Random Fourier Features:
# 

# In[3]:


df_mega.select_features("random_projection", n_components=10)

# **Random Fourier Features** the importance indicates how strongly each feature influences the approximation of non-linear relationships in the Fourier-transformed space.
# 

# In[4]:


df_mega.select_features("fourier", n_components=10)

# 
# **Independent Component Analysis (ICA):**
# The feature importance is based on the magnitude of each feature's contribution to the extracted independent components, which represent underlying independent signals in the data.
# 

# In[5]:


df_mega.select_features("ica", n_components=10)

# **Truncated Singular Value Decomposition (SVD):**
# The importance is determined by each feature's influence on the principal singular vectors, which represent directions of maximum variance in the data.

# In[6]:


df_mega.select_features("svd", n_components=10)

# **Sparse Random Projection:**
# The feature importance is based on how much each feature contributes to the variance in the sparsely projected space, similar to standard Random Projection but with improved computational efficiency.

# In[7]:


df_mega.select_features("sparse_projection", n_components=10)

# **Clustered SHAP Ensemble:** This method iteratively applies clustering, uses XGBoost to predict cluster membership, calculates SHAP values, and averages results across multiple runs to determine feature importance in identifying natural data structures.
# 

# In[8]:


df_mega.select_features("shapley", n_components=10)
