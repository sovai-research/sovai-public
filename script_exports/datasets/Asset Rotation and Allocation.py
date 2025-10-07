#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Asset Rotation & Allocation

# This dataset is perhaps better viewed through a dashboard, please use your login details to acces sit here.

# **All Risks**

# In[3]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

## addition

# In[4]:


df_allocate = sov.data("allocation/all"); df_allocate

# In[ ]:


df_past = sov.data("allocation/past"); df_past 

# In[6]:


df_future = sov.data("allocation/future"); df_future

# In[7]:


df_returns = sov.data("allocation/returns"); df_returns

# In[8]:


sov.plot("allocation", "line")

# In[8]:


sov.plot("allocation", "stacked")
