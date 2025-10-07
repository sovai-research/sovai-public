#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Core Economic Data

# In[2]:


# Macro Features are Dynamic (Let's lt it run then add parquet)

# In[4]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[6]:


sov.update_data_files()

# In[5]:


df_macro = sov.data("macro/features")

# In[4]:


df_macro

# In[3]:


df_macro
