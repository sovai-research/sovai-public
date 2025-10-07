#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Bankruptcy Prediction

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

# The bankruptcy prediction model is built to enchance equity selection, currently we have five different machine learning models that have a bankruptcy prediction accuracy from around 89% and [ROC-AUC](https://www.geeksforgeeks.org/auc-roc-curve/) of 85%. 
# 
# <span style="color:violet;">Requirement: Large computer with at least 16GB of RAM (Memory)</span> 
# 
# 

# To authenticate the instance, just use your email and password

# In[2]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# ## Bankruptcy Commands
# 
# There are a list of bankruptcy commands in the formats below, we will use them in this notebook.
# 
# * ##### **`sov.data('query')`**
# * ##### **`sov.plots('query')`**
# * ##### **`sov.reports('query')`**
# * ##### **`sov.compute('query')`**

# ### Datasets

# #### Probability
# Monthly bankruptcy probability `'bankruptcy/monthly'`, there is also a daily version `bankruptcy/daily` but it doesn't contain historical data 

# In[3]:


import sovai as sov
df_bankrupt = sov.data('bankruptcy', tickers=["MSFT","TSLA","META"]); df_bankrupt.tail()

# In[4]:


df_bankrupt.query("ticker == 'MSFT'").tail()

# In[5]:


df_bankrupt

# The daily predictions (`bankruptcy/daily`) series does not have a long history of predictions

# In[6]:


import sovai as sov
df_bankrupt_daily = sov.data('bankruptcy/daily', tickers=["MSFT","TSLA","META"], purge_cache=True); df_bankrupt_daily.tail()

# The probablilty you see here is from 0-100, so below 1 is a very low and negligible likelihood of bankruptcy. 

# In[7]:


df_bankrupt_daily.ticker("TSLA").plot_line()

# #### Feature Importance
# Shapley local average feature importance values `bankruptcy/shapleys` for prediction models.

# In[8]:


df_importance = sov.data('bankruptcy/shapleys', tickers=["MSFT","TSLA","META"]); df_importance.tail()

# #### Exploring Data
# Let's see the average importance that free cash flow has had over the years in explaining bankruptcies.

# In[9]:


df_importance.abs().groupby("date").mean()["fcf"].plot()

# The plot quantitatively shows that cashflow `"fcf"` bankruptcies were likely to become the main cause of public company bankruptcies in 2015. Eight of the ten largest public company bankruptcies that year were from oil, gas, and coal companies. In 2015, these industries faced significant difficulties due to a sharp drop in commodity prices. Although having substantial assets like reserves or equipment, these companies couldn't quickly convert them to cash to cover immediate obligations. This lack of liquidity forced many to file for bankruptcy protection.
# 
# **We can use any of the countless features.**

# In[10]:


df_importance.columns

# In[11]:


df_importance.abs().groupby("date").mean()["debt"].plot()

# ### Reports

# This **will take long** as we are downloading all the data. This report is a summary of the predicted bankruptcies.

# In[12]:


sov.report("bankruptcy", report_type="ranking")

# In[18]:


sov.report("bankruptcy", report_type="change")

# We can then also look at the **month-to-month** change

# ### Plots

# #### Bankruptcy Comparison

# In[14]:


sov.plot('bankruptcy', chart_type='compare', tickers=["TSLA","AMZN", "META", "CLMT"])

# #### Feature Importance over Time

# In[15]:


sov.plot("bankruptcy", chart_type="shapley", tickers=["TSLA"])

# #### Feature Importance Total

# In[16]:


sov.plot("bankruptcy", chart_type="stack", tickers=["DDD"])

# #### Bankruptcy vs Security Price

# In[17]:


sov.plot("bankruptcy", chart_type="line", tickers=["DDD"])

# #### PCA Statistical Similarity

# In[19]:


sov.plot("bankruptcy", chart_type="pca", tickers=["DDD"])

# #### Correlation Similarity

# In[20]:


sov.plot("bankruptcy", chart_type="similar", tickers=["DDD"])

# #### Similar Trends

# In[21]:


sov.plot("bankruptcy", chart_type="facet", tickers=["DDD"])

# #### **Global Charts**

# #### Bankruptcy Explanations

# In[22]:


sov.plot("bankruptcy", chart_type="time_global")

# #### Classification Performance - Confusion Matrix

# In[23]:


sov.plot("bankruptcy", chart_type="confusion_global")

# #### Classification Performance - Threshold Plots

# In[24]:


sov.plot("bankruptcy", chart_type="classification_global")

# #### Classification Performance - Lift Curve

# In[25]:


sov.plot("bankruptcy", chart_type="lift_global")

# ## Computations

# #### Similarity - Cause of Bankruptcy Risk (Caution: <span style="color:violet;"> Large Dataset </span>)
# 

# In[ ]:


df_importance = sov.data('bankruptcy/shapleys', full_history=True).set_index(["ticker","date"],drop=True).drop(columns=["version"])

# **Accross Tickers**

# This gives you a straightforward measure of how similar each pair of tickers is based on the cosine distance (**the smaller the distance, the more similar**)

# In[4]:


df_imp_distmat = df_importance.distance(on="ticker")

# In[5]:


df_imp_distmat["TSLA"].sort_values()

# `distance-cross-matrix` integrates how each ticker relates to all others in the dataset. This can reveal deeper, more complex relationships that might not be apparent from the direct cosine distances alone

# In[6]:


df_cross_distance = df_imp_distmat.relative_distance()

# In[7]:


df_cross_distance["TSLA"].sort_values()

# **What about across dates?** 
# 
# Allows us to identify how similar periods were according to driver of bankruptys

# In[8]:


df_imp_dist_dt = df_importance.distance(on="date")

# In[10]:


df_imp_dist_dt["2023-11-30"].sort_values()

# ### Similarity - Bankruptcy Probability

# In[20]:


df_prob = sov.data('bankruptcy', full_history=True).drop(columns=["version"])

# **Accross Tickers**

# In[21]:


df_prob_distmat = df_prob.distance(on="ticker")

# In[22]:


df_prob_distmat["SIVBQ"].sort_values()

# **Accross Dates**

# In[24]:


df_prob_distmat = df_prob.distance(on="date")

# In[25]:


df_prob_distmat["2023-11-30"].sort_values()

# **Percentile Calculation**

# In[26]:


df_importance_pct = df_importance.percentile(on="date")

# In[27]:


df_importance_pct.head()

# **Feature Mapping**

# In[28]:


df_importance_pct_mapped = sov.compute('map-accounting-features', df=df_importance_pct)

# In[29]:


df_importance_pct_mapped.head() 

# **Customized Advanced Example 1**

# In[30]:


df_importance_pca = df_importance_pct.pca(n_components=4) 
df_balanced_sampled, max_date, max_lag = sov.compute('process-pca-plot', df=df_importance_pca)
sov.plot('bankruptcy', chart_type="pca_clusters", df=df_balanced_sampled, target='target', max_lag=max_lag, max_date=max_date)

# **Customized Advanced Example 2**

# In[35]:


df_bankrupt = sov.data('bankruptcy', full_history=True)
df_bankrupt, sorted_tickers = sov.compute('process-bankrupt-plot', df=df_bankrupt)
sov.plot('bankruptcy', chart_type="predictions", df=df_bankrupt, sorted_tickers=sorted_tickers)
