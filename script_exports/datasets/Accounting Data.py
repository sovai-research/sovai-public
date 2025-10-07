#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Accounting Data

# #### Quick Tips

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

# #### Authenticate

# In[1]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Latest Data

# In[2]:


df_accounting = sov.data("accounting/weekly",  purge_cache=True)

# In[3]:


df_accounting

# In[5]:


df_accounting.filter(["cash_short_term > 10m","start with ticker A","negative profits" ])

# In[6]:


df_accounting.filter(["sector = health", "get the companies with bottom two decile bankruptcy and top two decile breakout prediction and top two decile overshorted"])

# In[7]:


df_accounting = sov.data("accounting/weekly",  purge_cache=True, tickers=["AAPL","TLSA","MSFT"]); df_accounting.tail()

# #### Tickers Select
# 
# You can select tickers or identifiers like cusip and openfigi

# In[5]:


df_accounting = sov.data("accounting/weekly", tickers=["MSFT", "TSLA", "BBG001S5N8T1", "AMZN", "MMM", "MCD","DDD","0000006201"]); df_accounting

# In[9]:


df_accounting.filter(["cash_short_term > 10m", "ebitda > 10b", "bankruptcy_prediction < 10%"])

# #### Processed Dataset

# In[8]:


df_accounting = sov.data("accounting/weekly", tickers=["MSFT", "TSLA", "META", "AMZN", "MMM", "MCD","DDD"]); df_accounting

# In[17]:


sov.plot("accounting/weekly", chart_type="balance", ticker="MSFT")

# ### Intelligent Balance Sheet Analysis
# 
# This AI report takes around 2 minutes to run.

# In[6]:


sov.report("accounting", report_type="balance_sheet", ticker="MSFT")

# ### Historical Data
# 
# These are generally very large files when **`full_history` equals `True`**, as it essentially grabs the entire dataset.

# In[ ]:


# df_accounting = sov.data("accounting/weekly", full_history=True); df_accounting

# **Fractional Difference Computation**

# In[7]:


df_accounting.fractional_difference()

# **Unsupervised Feature Importance**

# In[8]:


df_accounting.importance()

# **Unsupervised Feature Selection**

# In[9]:


df_accounting.select_features(variability=0.50)

# **Compound Processing**

# (1) Fractional differencing, (2) unsupervised feature selection using PCA clustering, (3) reorder columbs, (4) orthogonalization by importance, and (5) percentile rank calculation are applied to the DataFrame.
# 

# This processing pipeline is crucial for our accounting financial standardized database as it helps to:
# 
#     1. Remove long-term memory and noise through fractional differencing.
#     2. Select the most informative features using Random Matrix projection.
#     3. Reorder the columns for the most important from left to right.
#     4. Orthogonalize features to reduce multicollinearity.
#     5. Calculate percentile ranks for better comparability across different scales.
# 
# These steps enhance the quality and interpretability of the data, enabling more accurate and reliable financial analysis and modeling.

# In[11]:


df_accounting.fractional_difference().select_features().feature_importance().importance_reorder().orthogonalize_features().percentile()

# In[7]:


sov.plot("accounting", chart_type="cashflows", df=df_accounting)

# In[52]:


sov.plot("accounting", chart_type="assets", df=df_accounting)

# **Perform Accounting Nowcasting**

# In[ ]:


df_accounting.nowcast_data("total_revenue")

# In[ ]:


df_accounting.query("ticker=='TSLA'").nowcast_plot("total_revenue")

# In[50]:


df_accounting.plot_line("total_revenue")

# ### Accounting Distance Comparison

# In[17]:


df_accounting.distance(on="ticker")

# In[18]:


df_accounting.percentile("date").distance(on="ticker")

# In[19]:


df_accounting.relative_distance(on="ticker")

# #### Try generating code with `sov.code`

# In[20]:


import sovai as sov

df = sov.code("get 'accounting/weekly' data between 2020 and 2021 for apple, microsoft and five more similar stocks", run=True, verbose=True)

# In[21]:


df
