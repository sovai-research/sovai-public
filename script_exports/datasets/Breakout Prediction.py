#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Breakout Prediction

# You can run the following commands to retrieve data using `sov.data`:
# 
# To fetch the **latest data** for a specific query:
# 
# ```python
# sov.data("query")
# ```
# 
# To fetch the **full historical data** for a specific query:
# 
# ```python
# sov.data("query", full_history=True)
# ```
# 
# To fetch the **full data** multiple **tickers** or identifiers like **cusip** and **openfigi**:
# 
# ```python
# sov.data("query", tickers=["9033434", "IB94343", "43432", "AAPL"])
# ```
# 
# To filter **any dataframe** just write some queries:
# 
# ```python
# df_accounting.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])
# ```
# 

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Grab historical predictions - Large File (2 mins)

# In[3]:


df_breakout = sov.data("breakout"); df_breakout

# Let's look at the latest data, although the model is explicity trained on the long-side, the short-side could also contain some positive signal.

# In[4]:


df_breakout.sort_values("slope")

# It is sometimes advised to remove the top and bottom 1% of the data as they could be related to noise or firms with no liquidity. 

# In[5]:


df_breakout = df_breakout[(df_breakout["prediction"] > df_breakout["prediction"].quantile(0.01)) & (df_breakout["prediction"] < df_breakout["prediction"].quantile(0.99))]

# In[6]:


df_breakout

# Let's plot a simple prediction over time.

# Let's also add some confidence intervals:
# (1) Change of slope is a **strong** indicator, (2) change of slope + above/below 50% is a **very strong** indicator. 

# In[7]:


df_msft = sov.data("breakout", start_date="2025-01-01")

# In[8]:


df_msft = sov.data("breakout", tickers=["MSFT"])

# In[9]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")
sov.plot("breakout", chart_type="predictions", df=df_msft)

# In[10]:


sov.plot("breakout", chart_type="accuracy", df=df_msft)
