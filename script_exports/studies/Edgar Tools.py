#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[1]:


# !pip install edgartools==2.27.5

# In[4]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[5]:


sov.sec_search("CFO Resignation")

# In[8]:


import pandas as pd
df = pd.read_csv("edgar_search_results/edgar_search_results_30092024_042718.csv")

# In[10]:


df.head()

# Let's explore some filings

# In[4]:


import sovai as sov
nflx_filing = sov.sec_filing("NFLX", "10-Q", "2022-06-06")

# In[5]:


nflx_filing.report

# In[6]:


nflx_filing.balance_sheet

# In[7]:


nflx_filing.income_statement

# In[8]:


nflx_filing.cash_flow_statement

# In[9]:


nflx_filing.sampled_facts

# In[17]:


nflx_filing.plot_facts
