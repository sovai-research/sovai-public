#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Liquidity Data

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

# In[1]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Price Improvement Data

# In[2]:


df_improve = sov.data("liquidity/price_improvement", tickers=["AAPL", "MSFT"])

# In[3]:


df_improve

# ### Market Opportunity

# In[4]:


df_market = sov.data("liquidity/market_opportunity", tickers=["AAPL", "MSFT"])

# In[5]:


df_market
