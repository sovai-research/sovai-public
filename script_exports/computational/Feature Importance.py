#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[12]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# Load ratios - takes around 5 mins to load data 
df_accounting = sov.data("accounting/weekly", start_date="2024-01-26")


# Choose long enough history for the model to train
df_mega = df_accounting.select_stocks("mega")

# ### Feature Importance

# **Random Projection:**
# The feature importance reflects how much each feature contributes to the variance in the randomly projected space.
# Random Fourier Features:
# 

# In[13]:


df_mega.importance("random_projection")

# **Random Fourier Features** the importance indicates how strongly each feature influences the approximation of non-linear relationships in the Fourier-transformed space.
# 

# In[14]:


df_mega.importance("fourier")

# 
# **Independent Component Analysis (ICA):**
# The feature importance is based on the magnitude of each feature's contribution to the extracted independent components, which represent underlying independent signals in the data.
# 

# In[15]:


df_mega.importance("ica")

# **Truncated Singular Value Decomposition (SVD):**
# The importance is determined by each feature's influence on the principal singular vectors, which represent directions of maximum variance in the data.

# In[16]:


df_mega.importance("svd")

# **Sparse Random Projection:**
# The feature importance is based on how much each feature contributes to the variance in the sparsely projected space, similar to standard Random Projection but with improved computational efficiency.

# In[17]:


df_mega.importance("sparse_projection")

# **Clustered SHAP Ensemble:** This method iteratively applies clustering, uses XGBoost to predict cluster membership, calculates SHAP values, and averages results across multiple runs to determine feature importance in identifying natural data structures.
# 

# In[18]:


df_mega.importance("shapley")

# ### Global Feature Importance

# In[19]:


df_mega.feature_importance()

# ### Feature Selection
# 
# An example of how you can select the top 25 features, using sparse projection. The feature selection process uses the importance scores to select the top features, reducing dimensionality while retaining the variables that have the most significant impact on the data's structure or variance.

# In[20]:


feature_importance = df_mega.importance("sparse_projection")

# In[21]:


df_select = df_mega[feature_importance["feature"].head(25)]; df_select
