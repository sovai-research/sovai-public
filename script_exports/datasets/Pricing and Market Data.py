#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Pricing and Market Data

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[3]:


sov.data("market/prices", "TSLA")

# In[7]:


tickers = ["MSFT","TSLA", "AMZN", "DDD", "JPM","GS","F","A"]
df_price = sov.data("market/prices", tickers)

# In[8]:


df_price.tail()

# #### Grab just closing prices

# In[3]:


tickers = ["MSFT","TSLA", "AMZN", "DDD", "JPM","GS","F","A"]

df_close = sov.data("market/closeadj", tickers)

# In[4]:


df_close

# In[5]:


df_close = sov.data("market/closeadj")

# In[6]:


df_close

# Let's add some technical indicators

# In[8]:


df_tech = df_price.technical_indicators()

# In[9]:


df_tech.tail()
