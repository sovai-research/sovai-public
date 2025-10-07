#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Quick start

# ### Authorization and Authentication

# In[4]:


import sovai as sv

# There are three ways how to login to the API

# 1. Configuration API connection
sv.ApiConfig.token = "super_secret_token"
sv.ApiConfig.base_url = "https://google.com"

# 2. Read token from .env file e.g API_TOKEN=super_secret_token
sv.read_key('.env')

# 3. The Basic authentication method
sv.basic_auth("test@test.com", "super_strong_password")

# And then continue working with get some data from API and manipulating them

# ### Retrieve data from different endpoints from the API server

# In[5]:


gs_df = sv.get("bankruptcy/monthly", params={"version": 20221013})
gs_df.head()

# ### Retrieve charts data with plotting graphs

# In[13]:


# from IPython.display import Image, display
# Retrieve data with plotting special flag `plot=True`
data_pca = sv.get(
    endpoint="bankruptcy/charts", params={"tickers": "A", "chart": "pca"}, plot=True
)
