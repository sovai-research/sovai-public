#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Anomaly Computation

# Global anomalies look at the entire dataset, local anomalies focus on neighborhoods, and cluster anomalies consider the multi-dimensional structure of the data.

# In[3]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# ### Multivariate Anomaly Detection - Accounting Factors

# In[11]:


# Load ratios - takes around 
df_factors = sov.data("factors/accounting", purge_cache=True, start_date="2021"); df_factors.head()

# #### Local, Global, and Cluster Scores (For NVDA)

# I only want to look at the last **three years**, the data should be in **percentile format**

# In[12]:


import pandas as pd

ticker = "TSLA"

df_last_3_years = df_factors.loc[(slice(None), slice(pd.Timestamp.now() - pd.DateOffset(years=3), None)), :]

df_last_3_years = df_last_3_years.percentile()

df_anomaly_scores = df_last_3_years.anomalies("scores",ticker)

# In[13]:


df_anomaly_scores

# In[14]:


df_anomaly_scores.plot_line("local_anomaly_score", n=50)

# In[15]:


df_anomaly_scores.query("ticker ==@ticker").reset_index().set_index(["date"]).drop(columns=['ticker']).plot()

# In[16]:


import plotly.express as px

px.area(df_anomaly_scores.query("ticker == 'NVDA'").reset_index().set_index("date").drop(columns=['ticker']), title="NVDA Anomaly Scores Over Time", labels={"value": "Anomaly Score", "variable": "Feature"}, line_shape="spline").show()

# In[17]:


import plotly.express as px

px.bar(df_anomaly_scores.query("ticker == 'NVDA'").reset_index().melt(id_vars=['date', 'ticker'], var_name='Feature', value_name='Anomaly Score'), x='date', y='Anomaly Score', color='Feature', title="NVDA Anomaly Scores Over Time", barmode='stack').show()

# In[18]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_anomaly_dashboard(df_anomaly_scores, ticker):
    # Filter for the specific ticker and calculate percentile ranks
    df_ticker = df_anomaly_scores.loc[ticker].reset_index()
    df_all = df_anomaly_scores.reset_index()
    
    for score in ['global_anomaly_score', 'local_anomaly_score', 'clustered_anomaly_score']:
        df_ticker[f'{score}_percentile'] = df_ticker[score].rank(pct=True)
    
    # Calculate ratios and interactions
    for df in [df_ticker, df_all]:
        df['global_local_ratio'] = df['global_anomaly_score'] / df['local_anomaly_score']
        df['global_clustered_ratio'] = df['global_anomaly_score'] / df['clustered_anomaly_score']
        df['local_clustered_ratio'] = df['local_anomaly_score'] / df['clustered_anomaly_score']
        df['interaction_score'] = df['global_anomaly_score'] * df['local_anomaly_score'] * df['clustered_anomaly_score']
    
    # Create subplots
    fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=('Anomaly Scores Over Time', 'Anomaly Score Percentiles',
                                        'Anomaly Score Ratios', 'Interaction Score',
                                        'Global vs Local Anomaly', 'Global vs Clustered Anomaly'))
    
    # Anomaly Scores Over Time
    for score, color in zip(['global_anomaly_score', 'local_anomaly_score', 'clustered_anomaly_score'], ['blue', 'red', 'green']):
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker[score], name=score.split('_')[0].capitalize(), line=dict(color=color)), row=1, col=1)
    
    # Anomaly Score Percentiles
    for score, color in zip(['global_anomaly_score', 'local_anomaly_score', 'clustered_anomaly_score'], ['blue', 'red', 'green']):
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker[f'{score}_percentile'], name=f'{score.split("_")[0].capitalize()} Percentile', line=dict(color=color)), row=1, col=2)
    
    # Anomaly Score Ratios
    for ratio, color in zip(['global_local_ratio', 'global_clustered_ratio', 'local_clustered_ratio'], ['purple', 'orange', 'brown']):
        fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker[ratio], name=ratio.replace('_', '/'), line=dict(color=color)), row=2, col=1)
    
    # Interaction Score
    fig.add_trace(go.Scatter(x=df_ticker['date'], y=df_ticker['interaction_score'], name='Interaction', line=dict(color='pink')), row=2, col=2)
    
    # Global vs Local Anomaly
    fig.add_trace(go.Scatter(x=df_all['global_anomaly_score'], y=df_all['local_anomaly_score'], mode='markers', name='All Tickers', marker=dict(color='lightgrey', size=5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ticker['global_anomaly_score'], y=df_ticker['local_anomaly_score'], mode='markers', name=ticker, marker=dict(color='red', size=8)), row=3, col=1)
    
    # Global vs Clustered Anomaly
    fig.add_trace(go.Scatter(x=df_all['global_anomaly_score'], y=df_all['clustered_anomaly_score'], mode='markers', name='All Tickers', marker=dict(color='lightgrey', size=5)), row=3, col=2)
    fig.add_trace(go.Scatter(x=df_ticker['global_anomaly_score'], y=df_ticker['clustered_anomaly_score'], mode='markers', name=ticker, marker=dict(color='red', size=8)), row=3, col=2)
    
    # Update layout
    fig.update_layout(height=1200, title_text=f"Anomaly Score Analysis Dashboard for {ticker}")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_xaxes(title_text="Global Anomaly Score", row=3, col=1)
    fig.update_xaxes(title_text="Global Anomaly Score", row=3, col=2)
    fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_yaxes(title_text="Percentile", row=1, col=2)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="Interaction Score", row=2, col=2)
    fig.update_yaxes(title_text="Local Anomaly Score", row=3, col=1)
    fig.update_yaxes(title_text="Clustered Anomaly Score", row=3, col=2)
    
    return fig

# Usage
ticker = "NVDA"  # or any other ticker in your dataset
fig = create_anomaly_dashboard(df_anomaly_scores, ticker)
fig.show()

# ### Local, Global, and Cluster Anomalies (Feature-Level)

# Here we can also analyse at a feature level what causes the anomalies for the security-date combinations.

# In[19]:


def most_anomalous_features(df, ticker, years=3):
    """
    Identify and sort the most anomalous features for a given ticker.
    
    Parameters:
    df (DataFrame): The input DataFrame containing all data.
    ticker (str): The ticker symbol to analyze.
    years (int): The number of recent years to consider (default is 3).
    
    Returns:
    DataFrame: Sorted anomalous features for the latest date of the specified ticker.
    """
    # Get data for the last 3 years and calculate local anomalies

    # Filter data for the specified ticker
    ticker_data = df.query("ticker == @ticker")
    
    # Get the latest date
    max_date = ticker_data.index.max()
    
    # Get data for the latest date and transpose
    latest_data = ticker_data.query('index == @max_date').T
    
    # Sort the features based on their anomaly scores
    if latest_data.columns.nlevels > 1:
        # For multi-level columns
        first_column = latest_data.columns[0]
    else:
        # For single-level columns
        first_column = latest_data.columns[0]
    
    sorted_result = latest_data.sort_values(by=first_column, ascending=False)
    
    return sorted_result


# In[20]:


# Local Most Anomalous Features
df_local = df_last_3_years.anomalies("local", ticker=ticker); df_local.head()

# In[21]:


result = most_anomalous_features(df_local, ticker='NVDA'); result

# In[22]:


## Global Most Anomalous Features
df_global = df_last_3_years.anomalies("global", ticker=ticker); df_global.head()

# In[23]:


result = most_anomalous_features(df_global, ticker='NVDA'); result

# In[25]:


df_last_3_years

# In[26]:


## Clustered Most Anomalous Features
df_cluster = df_last_3_years.anomalies("cluster", ticker=ticker); df_cluster.head()

# In[27]:


result = most_anomalous_features(df_cluster, ticker='NVDA'); result

# In[28]:


df_full = (df_local + df_global + df_cluster)/3

# In[29]:


result = most_anomalous_features(df_full, ticker='NVDA'); result

# In[30]:


df_full[[result.reset_index()["index"].values[0][0]]].query("ticker == @ticker").plot_line()

# In[31]:


df_last_3_years[[result.reset_index()["index"].values[0][0]]].query("ticker == @ticker").plot_line()

# ### Reconstruction Anomaly

# The positive and negative reconstruction error can give us the direction of the anomaly

# In[32]:


df_recons = df_factors.anomalies("reconstruction", "TSLA")

# In[33]:


df_recons.head()
