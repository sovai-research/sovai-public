#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Factor Model

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

# #### Processed Dataset

# In[3]:


%%time
df_accounting = sov.data("factors/accounting", tickers=["TSLA","MSFT","AAPL"], purge_cache=True); df_accounting.tail()

# ### Composite Score

# In[4]:


weights = {
    'profitability': 0.15,
    'value': 0.15,
    'solvency': 0.1,
    'cash_flow': 0.1,
    'momentum': 0.1,
    'earnings_consistency': 0.05,
    'small_size': 0.05,
    'low_equity_issuance': 0.05,
    'accrual_growth': 0.05,
    'current_liquidity': 0.05,
    'low_rnd': 0.05,
    'illiquidity': 0.025,
    'short_term_reversal': 0.025,
    'price_volatility': 0.025,
    'dividend_yield': 0.025,
    'low_growth': 0.025,
    'bounce_dip': 0.025,
    'low_depreciation_growth': 0.025,
}

def composite_score(row):
    score = 0
    for factor, weight in weights.items():
        score += row[factor] * weight
    return score

df_accounting['composite_score'] = df_accounting.apply(composite_score, axis=1)

# Depending on the investors believe the above composite score can be constructed. 
# 
#     1. Profitability (+): More profitable firms tend to outperform less profitable firms, as they have a stronger financial position and better prospects for future growth.
#     
#     2. Value (+): Undervalued stocks (high book-to-market ratio) tend to outperform overvalued stocks, as they may be mispriced by the market and have more potential for price appreciation.
#     
#     3. Solvency (+): Firms with strong balance sheets and low debt ratios tend to outperform those with weaker financial positions, as they are less risky and more resilient during market downturns.
# 
# 
# *It's important to note that the strength and consistency of these relationships may vary across different markets, time periods, and economic conditions.*

# In[5]:


df_accounting.plot_line('composite_score')

# ### Comprehensive Factors
# 
# Addition of non-traditional financial metrics such as market risk, business risk, political risk, and inflation risk, this dataset helps in evaluating external factors that could impact a company's performance.

# In[6]:


df_comprehensive = sov.data("factors/comprehensive", tickers=["TSLA","MSFT"]); df_comprehensive.tail()

# In[7]:


df_comprehensive.plot_line("inflation_persistence")

# ### Factor Statistical Analysis

# #### Coefficient Values
# 
# 
# * **Purpose**: The coefficient values represent the estimated impact of each factor on the stock returns. They indicate the direction (positive or negative) and magnitude of the relationship between the factors and returns.
# * **Investor Benefit**: By tracking the coefficient values over time, investors can identify which factors have a significant influence on stock returns and how their impact changes. This information can help in making investment decisions based on the prevailing market conditions and factor exposures.
# 

# In[8]:


df_coefficients = sov.data("factors/coefficients", tickers=["TSLA","MSFT"]); df_coefficients.tail()

# #### Standard Errors
# 
# - **Purpose:** Standard errors measure the statistical uncertainty associated with the estimated coefficients. They provide a range around the coefficient estimates, indicating the precision of the estimates.
# - **Investor Benefit:** Lower standard errors suggest more precise coefficient estimates, increasing confidence in the factor relationships. Investors can assess the reliability of the factor models by monitoring the standard errors over time.
# 

# In[9]:


df_standard_errors = sov.data("factors/standard_errors", tickers=["TSLA","MSFT"]); df_standard_errors.tail()

# #### T-Statistics
# 
# - **Purpose:** T-statistics are used to determine the statistical significance of the coefficient estimates. They measure how many standard deviations the coefficients are from zero, helping to assess whether the factor relationships are statistically meaningful.
# - **Investor Benefit:** Significant t-statistics (usually greater than 2 or less than -2) indicate that the factor relationships are unlikely to be due to chance. Investors can focus on factors with consistently significant t-statistics, as they are more likely to have a real impact on stock returns.
# 

# In[10]:


df_t_statistics = sov.data("factors/t_statistics", tickers=["TSLA","MSFT"]); df_t_statistics.tail()

# #### Model Metrics
# 
# - R-squared and Adjusted R-squared:
#   - Purpose: R-squared measures the proportion of the variance in stock returns explained by the factor model. Adjusted R-squared accounts for the number of factors used, penalizing model complexity.
#   - Investor Benefit: Higher R-squared values suggest that the factor model captures a significant portion of the return variability. Investors can compare R-squared across models and over time to assess their explanatory power and identify the most effective models (generally R-Squared greater than 0.3).
# 
# - F-value:
#   - Purpose: The F-value evaluates the overall significance of the factor model. It tests whether the factors jointly have a significant impact on stock returns.
#   - Investor Benefit: A significant F-value indicates that the factor model as a whole is statistically meaningful. Investors can use the F-value to gauge the overall relevance of the model in explaining stock returns (generally F-value greater than 1).
# 

# In[11]:


df_model_metrics = sov.data("factors/model_metrics", tickers=["TSLA","MSFT"]); df_model_metrics.tail()
