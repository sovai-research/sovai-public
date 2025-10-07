#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Insider Trading

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

# In[5]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[10]:


df_insider = sov.data("insider/flow_prediction", full_history=True)

# In[12]:


df_insider

# In[13]:


df_insider = sov.data("insider/trading", frequency="difference")

# In[20]:


df_insider.sort_values("market_impact")

# In[15]:


df_insider = sov.data("insider/trading"); df_insider.sort_values("flow_prediction")

# In[6]:


sov.plot("insider", chart_type="prediction")

# Double click on a line or legend below the plot to isolate a series

# In[8]:


sov.plot("insider", chart_type="flows")

# Features and Predicted Flow Patterns

# In[9]:


sov.plot("insider", chart_type="percentile", ticker="AAPL")
