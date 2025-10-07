#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Lobbying Analysis

# In[2]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[3]:


import sovai as sov
import pandas as pd
import numpy as np
import ast
import datetime

sov.token_auth(token="visit https://sov.ai/profile for your token")

df_lobbying = sov.data("lobbying",verbose=False, start_date=datetime.datetime.now() - datetime.timedelta(days=50))

df_lobbying


# #### Processed Dataset

# In[5]:


df_lobbying = sov.data("lobbying", tickers=["PRIVATE"], start_date="2024-02-04", purge_cache=True)

# In[6]:


df_lobbying

# In[7]:


%time
df_lobbying = sov.data("lobbying", tickers=["WFC","EXPGY","AAPL"], purge_cache=True)

# In[8]:


df_lobbying.sort_values("date")
