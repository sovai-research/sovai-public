#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Short Data

# You can run the following commands to retrieve data (`df`) using `sov.data`:
# 
# To fetch the **latest data** for a specific query:
# 
# ```python
# df = sov.data("query")
# ```
# 
# To fetch the **full historical data** for a specific query:
# 
# ```python
# df = sov.data("query", full_history=True)
# ```
# 
# To fetch the **full data** multiple **tickers** or identifiers like **cusip** and **openfigi**:
# 
# ```python
# df = sov.data("query", tickers=["9033434", "IB94343", "43432", "AAPL"])
# ```
# 
# To filter **any dataframe** just write some queries:
# 
# ```python
# df.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])
# ```
# 

# In[2]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

df_short_vol_hist = sov.data("short/volume", full_history=True)

# In[3]:


df_short_vol_hist

# ### Short Interest & Overshorted

# In[4]:


df_short = sov.data("short/over_shorted", tickers=["AAPL", "MSFT"])

# In[5]:


df_short.get_latest()

# **Latest Data Full:**

# In[6]:


df_short_all = sov.data("short/over_shorted")

# In[7]:


df_short_all.sort_values("over_shorted")

# ### Short Volume

# In[24]:


df_short_volume = sov.data("short/volume", tickers=["AAPL","AMZN", "META", "MSFT"])

# In[25]:


df_short_volume

# In[27]:


df_short_volume.plot_line("short_volume")
