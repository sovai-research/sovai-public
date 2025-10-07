#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Earnings Surprise Prediction

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


## To check
import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Earnings Surprise

# In[3]:


df_earn_surp = sov.data("earnings/surprise", tickers=["AAPL", "MSFT"])

# In[4]:


df_earn_surp

# In[5]:


df_earn = sov.data("earnings/surprise"); df_earn.sort_values('surprise_probability')

# In[8]:


sov.plot("earnings/surprise", "tree")

# In[9]:


sov.plot("earnings/surprise", "line")
