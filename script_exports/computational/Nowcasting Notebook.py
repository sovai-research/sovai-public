#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Nowcasting Data

# In[2]:


## To check
import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[10]:


df_accounting = sov.data("accounting/weekly", start_date="2020-01-26").select_stocks("mega")

# In[11]:


df_accounting.tail() 

# **Nowcasting** for a particular stock

# In[12]:


df_accounting.query("ticker == 'AAPL'").nowcast_data("accounts_receivable")

# **Nowcasting** for all stocks

# In[13]:


df_accounting.nowcast_data("accounts_receivable")

# In[16]:


df_accounting.nowcast_plot("total_revenue")
