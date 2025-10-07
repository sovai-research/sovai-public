#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Institutional Holdings

# In[3]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[4]:


df_institute = sov.data("institutional/trading", start_date="2004-04-30", tickers=["MSFT","TSLA", "AAPL","META","NFLX", "DDD"])

# In[5]:


df_institute.tail()

# #### Flow Prediction

# In[6]:


df_prediction = sov.data("institutional/flow_prediction", purge_cache=True)

# In[7]:


df_prediction

# In[8]:


df_prediction.sort_values("flow_prediction")

# In[9]:


df_prediction = sov.data("institutional/flow_prediction", frequency="difference")

# In[10]:


df_prediction.head()

# **Ranking Report:** Sort by Ascending to get low-flow prediction Stocks:

# In[11]:


sov.report("institutional/flow_prediction", report_type="ranking")

# #### Institutional Prediction Plot
# 
# ``Predicted Flows`` is good for forming trading startegies. For the first round it takes around 2 mins to download and process the data. 

# In[12]:


sov.plot("institutional", chart_type="prediction")

# ### Advanced Study
# Let's take a look at the ``predicted flows`` and ``actual flows`` over time. For the first round it takes around **``2 mins``** to download and process the data.

# In[16]:


sov.plot("institutional", chart_type="flows")
