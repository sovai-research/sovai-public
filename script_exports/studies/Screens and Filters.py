#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ### Screens and Filters
# 
# You can use any of the 100s of features as filtering and selection criteria for identifying novel investment opportunities. 

# In[1]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[2]:


df_comprehensive = sov.data("factors/comprehensive")

# Let's focus on the sensitivity of firms on business_risk indicators, high value means they move in close conjunction with business risks. 

# In[8]:


df_risk = df_comprehensive.get_latest("business_risk")[["business_risk"]]; df_risk

# There are a lot of small firms in this list, I would like to isolate only large caps.

# In[10]:


df_mega = df_risk.select_stocks("mega"); df_mega

# From the large cap's I would prefer to select those with business risk sensitivity below 10 percent. 

# In[12]:


df_ten = df_mega.query("business_risk <= 10")

# In[13]:


df_ten

# And of course all of this could be written in one line! 

# In[15]:


df_ten = sov.data("factors/comprehensive").get_latest("business_risk")[["business_risk"]].select_stocks("mega").query("business_risk <= 10"); df_ten
