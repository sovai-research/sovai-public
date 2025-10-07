#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Segmentation Notebook

# In[8]:


## To check
import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Change Point Detection

# In[9]:


df_accounting = sov.data("accounting/weekly", start_date="2020-01-01").select_stocks("mega")

# In[10]:


df_accounting

# In[11]:


df_change = df_accounting.change_point(method='data', feature="book_equity_value"); df_change.tail(10)

# In[12]:


df_change.attrs['stats']

# In[13]:


df_accounting.change_point(method='plot')

# ### Regime Change

# In[14]:


rc_result = df_accounting.regime_change(method="data", ticker="AAPL", feature="total_revenue"); rc_result.tail(10)

# In[15]:


rc_result.attrs['stats']

# In[16]:


df_accounting.regime_change(method="plot", ticker="AAPL", feature="total_revenue")


# ### Comprehensive Regime

# In[17]:


pca_rc_result = df_accounting.pca_regime_change(method="data", ticker="AAPL"); pca_rc_result.tail()

# In[18]:


pca_rc_result.attrs['stats']

# In[19]:


df_accounting.pca_regime_change(method="plot", ticker="AAPL")
