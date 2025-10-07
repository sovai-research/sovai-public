#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Consumer Financial Complaint

# In[8]:


import sovai as sov

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[9]:


df_complaints = sov.data("complaints/public", tickers=["WFC","EXPGY"]); df_complaints.tail()

# In[10]:


import pandas as pd
import plotly.express as px

# Assuming your DataFrame is named 'wells_df'
# Convert 'date_received' column to datetime if not already

df_complaints = df_complaints.reset_index()

wells_df = df_complaints[df_complaints["ticker"]=="WFC"]

wells_df['date'] = pd.to_datetime(wells_df['date'])

# Resample the data by week and calculate the mean of 'total_risk'
weekly_data = wells_df.resample('M', on='date')['complaint_score'].mean().reset_index()

# Create the line plot using Plotly Express
fig = px.line(weekly_data, x='date', y='complaint_score', title='Weekly Mean Total Risk')

# Update the axis labels
fig.update_xaxes(title='Date')
fig.update_yaxes(title='Mean Total Risk')

# Show the plot
fig.show()

# In[11]:


df_complaints = sov.data("complaints/public"); df_complaints.tail()
