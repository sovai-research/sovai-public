#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ### Congressional Trading

# In[2]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[6]:


df_congress = sov.data("congress").sort_values("date")

# In[7]:


df_congress.tail()
