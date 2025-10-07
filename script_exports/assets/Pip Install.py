#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[1]:


# pip uninstall sovai -y && pip install -e "../.."

# In[10]:


# !pip install /Users/dereksnow/Sovai/GitHub/SovAI/dist/sovai-0.1.13-py3-none-any.whl

# In[ ]:


## first just make build for local version

# In[2]:


pip uninstall sovai -y

# In[3]:


!pip install /Users/dereksnow/Sovai/GitHub/SovAI/dist/sovai-0.2.11-py3-none-any.whl

# In[4]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

df_breakout = sov.data("breakout")


# In[5]:


df_breakout.get_latest()

# In[6]:


df_breakout.query("ticker =='AAPL'")
