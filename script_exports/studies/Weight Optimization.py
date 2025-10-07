#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[2]:


# !pip install skfolio

# In[3]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Data Preparation
# Let's prepare our dataset for weight optimization

# In[4]:


import pandas as pd

df_price = sov.data("market/closeadj", full_history=True)
df_mega = df_price.select_stocks("mega").date_range("2000-01-01")

# From long to wide format

# In[5]:


df_returns = df_mega.calculate_returns().dropna(axis=1,how="any")

# There are more than 50 mega companies with long histories

# In[6]:


df_returns.head(3)

# I only want to select 25 companies that are the most uncorrelated.

# In[7]:


feature_importance = df_returns.importance()

# In[8]:


feature_importance.head(5)

# In[9]:


df_select = df_returns[feature_importance["feature"].head(25)]

# ### Weight Optimization

# This step take 5 minutes to test 10 different machine learning for asset management models and configurations, and returns the best models plus an equal weighted benchmark. 

# In[10]:


portfolio = df_select.weight_optimization()

# 
# **Sharpe Ratio Distribution**
# `portfolio.sharpe_plot`
# Shows the distribution of Sharpe ratios across different strategies, helping to understand the consistency of risk-adjusted returns.
# 

# In[11]:


portfolio.sharpe_plot

# 
# **Cumulative Returns Plot**
# `portfolio.return_plot`
# Displays the cumulative returns of all portfolio strategies over time, allowing for easy comparison of overall performance.
# 

# In[12]:


portfolio.return_plot

# 
# **Overall Composition Plot**
# `portfolio.composition_plot`
# Illustrates the asset allocation of all strategies, allowing for a comparison of how different models allocate capital.
# 

# In[13]:


portfolio.composition_plot

# 
# **Best Performing Model**
# `portfolio.best_model`
# Identifies the strategy that performed best according to the Sharpe ratio.
# 

# In[14]:


portfolio.best_model

# 
# **Performance Summary**
# `portfolio.performance_report`
# Provides a comprehensive summary of key performance metrics for all strategies, including returns, volatility, Sharpe ratio, and more.
# 

# In[15]:


portfolio.performance_report

# ### Best Model Analytics

# In[16]:


best_model = portfolio.best_model; best_model

# 
# **Cumulative Returns**
# `portfolio["model_name"].backtest_plot`
# Displays the cumulative returns of the specific model over the backtesting period.
# 

# In[17]:


portfolio[best_model].backtest_plot

# 
# **Backtest Report**
# `portfolio["model_name"].backtest_report`
# Detailed performance statistics from the backtesting period for the specific model.
# 

# In[18]:


portfolio[best_model].backtest_report

# 
# **Rolling Sharpe Ratio**
# `portfolio["model_name"].sharpe_rolling_plot`
# Visualizes how the Sharpe ratio of the model changes over time, indicating consistency of performance.
# 

# In[19]:


portfolio[best_model].sharpe_rolling_plot

# 
# `portfolio["model_name"].composition_plot`
# Illustrates the asset allocation for the specific model.
# 

# In[20]:


portfolio[best_model].composition_plot

# 
# **Drawdown Contribution**
# `portfolio["model_name"].drawdown_contribution_plot`
# Shows which assets contribute most to the portfolio's drawdowns, helping identify risk sources.
# 

# In[21]:


portfolio[best_model].drawdown_contribution_plot

# 
# **Sharpe Ratio Contribution**
# `portfolio["model_name"].sharpe_contribution_plot`
# Indicates which assets contribute most to the portfolio's Sharpe ratio, highlighting return drivers.
# 

# In[22]:


portfolio[best_model].sharpe_contribution_plot

# 
# **Correlation Heatmap**
# `portfolio["model_name"].heatmap_plot`
# Displays the correlation structure of assets used in the model (not available for EQUAL).
# 

# In[23]:


portfolio[best_model].heatmap_plot

# 
# **Clustering Dendrogram**
# `portfolio["model_name"].cluster_plot`
# Visualizes the hierarchical clustering of assets used in the model (not available for EQUAL).
# 

# In[24]:


portfolio[best_model].cluster_plot

# 
# **Current Recommended Allocation**
# `portfolio["model_name"].recommended_allocation`
# Provides the model's most recent recommended asset allocation.
# 

# In[25]:


portfolio[best_model].recommended_allocation

# **Sharpe Ratio Distribution**
# `portfolio["model_name"].sharpe_dist_plot`
# Shows the distribution of Sharpe ratios across different strategies, helping to understand the consistency of risk-adjusted returns.
# 

# In[26]:


portfolio[best_model].sharpe_dist_plot

# 
# **Daily Weights**
# `portfolio["model_name"].daily_weights`
# Shows how the model's asset allocation changes day-by-day over the backtesting period.

# In[27]:


portfolio[best_model].daily_weights
