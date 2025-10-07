#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ### Signal Evaluation
# 
# Evaluate the quality of a proposed alpha trading signal.

# In[11]:


import sovai as sov
 
sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[12]:


df_comprehensive = sov.data("factors/comprehensive", full_history=True)

# In[13]:


df_comprehensive.tail()

# Let's focus on the sensitivity of firms on `business_risk` indicators, high value means they move in close conjunction with business risks. 

# In[14]:


df_risk = df_comprehensive[["business_risk"]]; del df_comprehensive; df_risk

# ## Alpha Signal Evaluation

# In[26]:


from sovai import data

df_closeadj = data("market/closeadj", full_history=True).rename({"closeadj": "price"}, axis=1)

# df_merge = self.merge(
# df_closeadj, left_index=True, right_index=True, how="left"
# )

# df_merge.isnull().sum()

# df_merge["price"] = df_merge.groupby(level="ticker")["price"].ffill()

# In[28]:


df_closeadj

# In[29]:


evaluator = df_risk.signal_evaluator(verbose=True)

# **Performance Plot:**
# This plot showcases the cumulative performance of the strategy over time, accompanied by a rolling Sharpe ratio. It displays the strategy returns, a 95% confidence interval derived from random simulations, and the rolling Sharpe ratio on a secondary y-axis.

# In[30]:


evaluator.performance_plot

# 
# **Signal Decile Plot:**
# This visualization presents the cumulative returns of the signal divided into deciles, illustrating how different strength levels of the signal perform over time. It also incorporates average Sharpe ratios for each decile.
# 

# In[31]:


evaluator.signal_decile_plot

# **Stress Test Plot:**
# This plot demonstrates how the strategy performs during various historical stress events or market crises, providing insights into the strategy's robustness during challenging market conditions.
# 

# In[32]:


evaluator.stress_plot

# 
# **Drawdown Plot:**
# This visualization captures the drawdowns of the strategy over time, highlighting periods of decline from previous peak values and aiding in the assessment of the strategy's risk profile.
# 
# 

# In[33]:


evaluator.drawdown_plot

# **Return Distribution Plot:**
# This plot illustrates the distribution of returns for the strategy, typically in the form of a histogram. It often includes key statistics such as mean return, standard deviation, and various risk metrics.
# 

# In[34]:


evaluator.distribution_plot

# 
# **Returns Heatmap:**
# This heatmap visualization displays the strategy's returns across different months and years, helping to identify any seasonal patterns in the strategy's performance.
# 

# In[35]:


evaluator.returns_heatmap_plot

# 
# **Signal Autocorrelation Plot:**
# This plot reveals the autocorrelation of the signal over time, offering insights into the signal's persistence and predictive power.
# 
# 

# In[36]:


evaluator.signal_correlation_plot

# **Portfolio Turnover Plot:**
# This visualization depicts the portfolio turnover over time, often separated into long and short positions, aiding in the assessment of trading costs and strategy stability.
# 
# 

# In[37]:


evaluator.turnover_plot

# **Performance Statistics Table:**
# This comprehensive table presents key performance statistics for the strategy, including metrics such as annualized returns, Sharpe ratio, maximum drawdown, and other relevant indicators.
# 

# In[38]:


evaluator.performance_table

# **Drawdown Analysis Table:**
# This detailed table enumerates the worst drawdown periods for the strategy, including their magnitude, duration, and recovery times.
# 

# In[39]:


evaluator.drawdown_table

# 
# ### Other Core Attributes
# 
# 1. **Position Holdings:** `evaluator.positions`
# 1. **Rebalance Schedule:** `evaluator.rebalance_mask`
# 1. **Actual Holdings:** `evaluator.holdings`
# 1. **Asset Returns:** `evaluator.returns`
# 1. **Position Returns:** `evaluator.position_returns`
# 1. **Rebalanced Returns:** `evaluator.resampled_returns`
# 1. **Portfolio Returns:** `evaluator.portfolio_returns`
# 1. **Cumulative Performance:** `evaluator.cumulative_returns`
