#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Time Decomposition Notebook

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[3]:


# Load ratios - takes around 5 mins to load data 
df_accounting = sov.data("accounting/weekly", start_date="2023-01-26").select_stocks("mega")

# In[5]:


df_time = df_accounting.time_decomposition(method="data", ticker="AAPL", feature="total_revenue"); df_time.tail()

# In[6]:


df_time.attrs["stats"]

# In[7]:


df_accounting.time_decomposition(method="plot", ticker="AAPL", feature="total_revenue")
