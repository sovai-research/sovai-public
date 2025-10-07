#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Corporate Risk Analysis

# You can run the following commands to retrieve data (`df`) using `sov.data`:
# 
# To fetch the **latest data** for a specific query:
# 
# ```python
# df = sov.data("query")
# ```
# 
# To fetch the **full historical data** for a specific query:
# 
# ```python
# df = sov.data("query", full_history=True)
# ```
# 
# To fetch the **full data** multiple **tickers** or identifiers like **cusip** and **openfigi**:
# 
# ```python
# df = sov.data("query", tickers=["9033434", "IB94343", "43432", "AAPL"])
# ```
# 
# To filter **any dataframe** just write some queries:
# 
# ```python
# df.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])
# ```
# 

# In[2]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### **Overall Risk**

# All historical data (takes 2 mins)

# In[3]:


df_risks = sov.data("corprisk/risks"); df_risks

# In[15]:


sov.data("corprisk/risks", tickers="AAPL", purge_cache=True)

# **Ranking Report:** Sort by Ascending to get low-risk Stocks:

# In[5]:


sov.report("corprisk/risks",report_type="ranking")

# In[6]:


sov.report("corprisk/risks",report_type="change")

# In[7]:


sov.plot("corprisk/risks",chart_type="line")

# ### Component Risks
# **Accounting Risks**

# In[8]:


df_actg_risk = sov.data("corprisk/accounting", full_history=True)

# If by mistake you loaded all the data, you can still perform post-hoc filtering:

# In[9]:


df_actg_risk.ticker("AAPL")

# In[12]:


sov.report("corprisk/accounting",report_type="ranking")

# In[13]:


sov.report("corprisk/accounting",report_type="change")

# **Event Risks**

# In[16]:


df_events_risk = sov.data("corprisk/events", full_history=True)

# In[14]:


sov.report("corprisk/events",report_type="ranking")

# In[17]:


df_events_risk.get_latest()

# **Misstatement Risk**

# In[23]:


df_miss_risk = sov.data("corprisk/misstatement")

# In[24]:


df_miss_risk.query("ticker == 'AAPL'")

# **Prinicpal Component Risks** (3 mins to run)

# In[25]:


import pandas as pd
cols_remove = ["average","industry_adj_avg"]
df_miss_risk = sov.data("corprisk/misstatement", full_history=True)

add_all = pd.concat([df_miss_risk.drop(columns=cols_remove), df_events_risk.drop(columns=cols_remove), df_actg_risk.drop(columns=cols_remove)],axis=1).sort_index().ffill().bfill()


# In[26]:


pca_df = add_all.pca(n_components=4)

# In[27]:


pca_df.query("ticker == 'CGRNQ'")

# A look at some structural statistical risks.

# In[ ]:


pca_df.groupby('date').median().plot()

# In[ ]:


last_month_pca = pca_df.loc[pca_df.index.get_level_values('date') == pca_df.index.get_level_values('date').max()]; last_month_pca.head()

# In[ ]:


last_month_pca.sort_values("PC4")
