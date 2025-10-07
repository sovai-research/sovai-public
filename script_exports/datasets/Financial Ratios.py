#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Financial Ratios

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

# In[3]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Standard Ratios

# In[4]:


df_ratios = sov.data("ratios/normal", tickers=["TSLA", "META","MSFT"], start_date="2024-02-20"); df_ratios.tail()

# In[29]:


df_ratios = sov.data("ratios/relative", tickers=["TSLA", "META","MSFT"], start_date="2024-02-20"); df_ratios.tail()

# In[13]:


df_ratiosdf_ratios

# In[9]:


df_summary = sov.data("ratios/relative",  tickers=["AAPL", "TSLA"], verbose=True, purge_cache=True)

# In[10]:


df_summary

# In[8]:


import pandas as pd
df = pd.read_parquet("https://storage.googleapis.com/sovai-public/concats/filters/latest.parquet")

# In[9]:


df

# In[11]:


df_summary

# In[17]:


df_summary.filter(["isdelisted =Y" ])

# In[4]:


df_summary

# In[5]:


df_ratios = sov.data("ratios/normal", tickers=["TSLA", "META"]); df_ratios.tail()

# In[6]:


df_ratios.get_latest()

# #### Relative Ratios

# In[5]:


df_percentile = sov.data("ratios/relative", start_date="2018-01-01", tickers=["TSLA", "META"])

# In[ ]:


df_summary.head()

# In[6]:


df_summary = sov.data("ratios/relative", frequency="latest")

# #### Vizualising Ratios

# In[7]:


import sovai as sov
sov.plot("ratios", chart_type="benchmark")

# In[8]:


sov.plot("ratios", chart_type="relative")
