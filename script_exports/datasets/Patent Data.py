#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ### Patent Data

# In[2]:


import polars as pl

# In[3]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[10]:


df_apps = sov.data("patents/applications", tickers=["AMZN"], start_date="2024-03-01", purge_cache=True)

# In[5]:


df_apps

# In[11]:


df_apps = sov.data("patents/applications",  start_date = "2025-04-01",  purge_cache=True)

# In[12]:


df_apps.head()
