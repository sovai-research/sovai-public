#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Clustering Module

# There are two types of clustering models, time-series, and cross-sectional. CS involves feature extraction and clustering algorithm. 

# In[38]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[43]:


# Load ratios - takes around 5 mins to load data 
df_accounting = sov.data("accounting/weekly", start_date="2023-01-26", purge_cache=True)

# In[44]:


df_accounting.shape

# In[45]:


df_accounting.head()

# In[46]:


# Choose long enough history for the model to train
df_mega = df_accounting.select_stocks("mega").date_range("2023-03-02")

# ### Panel Clustering
# 
# This clustering methodology takes multivariate panel datasets and represents them according to the centroids that capture the main patterns within the time series data. 

# #### All Features (5 minutes)
# First think we can do is to calculate the clusters according to all the features as compared accross all the tickers in the dataset. 

# In[47]:


df_mega.head()

# In[48]:


df_cluster = df_mega.cluster()

# In[49]:


df_cluster.tail()

# #### Specific Features
# We can also focus on any specific feature in the dataset of tens of features like `total_debt`, `total_assets`, or as we are using below `ebit`. 

# In[50]:


df_cluster_ebit = df_mega.cluster(features=["ebit"]); df_cluster_ebit.tail()

# We can also specify our own selection of multiple features like ``features=["total_assets","total_debt","ebit"]``.
# 

# In[51]:


df_mega.cluster(features=["total_assets","total_debt","ebit"])

# #### Downstream Calculations
# There are many things you can do once the data is clustered in time series, for one, you can take the standard deviation of the standard deviation of similarity accross clusters. 

# In[52]:


import pandas as pd

def transform_df(merged_df):
    # Reset the index to have 'ticker' and 'date' as columns
    df_reset = merged_df.reset_index()

    # Identify the numerical columns (assuming they start with 'Centroid')
    centroid_columns = [col for col in df_reset.columns if col.startswith('Centroid') and col != 'Centroid labels']

    # Calculate the average of the centroid columns
    df_reset['average'] = df_reset[centroid_columns].std(axis=1)

    # Pivot the table to have dates as index and tickers as columns
    transformed_df = df_reset.pivot(index='date', columns='ticker', values='average')

    return transformed_df

# Use the function
transformed_df = transform_df(df_cluster)
max_date = transformed_df.index.max()
sorted_df = (transformed_df.query("date == @max_date")
                           .T
                           .reset_index()
                           .sort_values(by=max_date, ascending=False)
                           .reset_index(drop=True))


# Companies who are stable and stay within their accounting cluster over time. 

# In[53]:


transformed_df.std().sort_values(ascending=False).tail(10)

# Companies who are see-sawing through accounting clusters over time. 

# In[54]:


transformed_df.std().sort_values(ascending=False).head(10)

# #### Distance Cluster
# 
# We can use our in-built distance functionality to get the distances between the ticker-cluster combinations.

# In[55]:


df_dist = df_cluster.drop(columns=["labels"]).distance(orient="time-series"); df_dist

# Distance calculation for companies with similar clusters:

# In[56]:


df_dist.sort_values(["AMZN"])[["AMZN"]].T

# What about ebit clustering distance?

# In[57]:


df_cluster_ebit.drop(columns=["labels"]).distance(orient="time-series").sort_values(["AMZN"])[["AMZN"]].T

# #### Summary
# This gives you a quick summary of the last 6-months data

# In[58]:


df_mega.cluster("summary")

# #### Vizualisation
# Each colored line represents a distance to centroid of the cluster. The centroid is the average pattern of all time series assigned to that cluster. These are similarity scores (based on cross-correlation). Selecting features shows you the different shapes over time. 

# In[60]:


df_mega.cluster("line_plot")

# In[61]:


df_mega.cluster("scatter_plot")

# In[62]:


df_mega.cluster("animation_plot")
