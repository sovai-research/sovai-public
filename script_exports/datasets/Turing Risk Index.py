#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Business, Political, and Market Risk (Turing Risk)

# This dataset is perhaps better viewed through a dashboard, please use your login details to acces sit here.

# **All Risks**

# In[6]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[8]:


df_risks = sov.data("risks",  purge_cache=True); df_risks.tail()

# In[10]:


# sov.data("risks/plot/turing_risk_plot", plot=True)

# In[11]:


df_risks = sov.data("risks"); df_risks.tail()

# In[16]:


df_risks[["MARKET_RISK","BUSINESS_RISK","POLITICAL_RISK","TURING_RISK"]].tail(15400).plot()

# In[17]:


df_risks[["RECESSION_6","RECESSION_12","RECESSION_24"]].tail(1000).plot()

# **Business Risks**

# In[13]:


df_business = sov.data("risks/business"); df_business.tail()

# **Political Risks**

# In[5]:


df_political = sov.data("risks/politics"); df_political.tail()

# **Market Risks**

# In[6]:


df_market = sov.data("risks/market"); df_market.tail()

# **Computations**

# In[19]:


df_risks_agg = sov.compute('risk-aggregates', df=df_risks); df_risks_agg.tail()

# In[20]:


df_risks_agg.plot_line("MANUFACTURING_SENTIMENT")

# In[21]:


df_risks.plot_line("TURING_RISK")
