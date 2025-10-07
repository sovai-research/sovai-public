#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Box Office Movies

# In[2]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[3]:


df_movies = sov.data("movies/boxoffice")

# In[4]:


df_movies.sort_values("date")

# In[6]:


import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

df = df_movies.reset_index().copy()
# Extract the year from the 'date' column
df = df.dropna(subset=["date"])
df['year'] = df['date'].dt.year

# Group by ticker and year, summing the gross
df_heatmap = df.groupby(['ticker', 'year'])['gross'].sum().reset_index()

# Replace NaN and infinite values with 0
df_heatmap['gross'] = df_heatmap['gross'].replace([np.inf, -np.inf, np.nan], 0)

# Group by ticker and sum the total gross
df_ticker_total = df_heatmap.groupby('ticker')['gross'].sum().reset_index()

# Sort the tickers by total gross in descending order
sorted_tickers = df_ticker_total.sort_values('gross', ascending=False)['ticker'].tolist()

# Create the pivot table
pivot_data = df_heatmap.pivot(index='ticker', columns='year', values='gross')

# Fill NaN values with 0
pivot_data = pivot_data.fillna(0)

# Select top 20 tickers by total gross
top_15_tickers = df_ticker_total.nlargest(15, 'gross')['ticker'].tolist()

# Filter the pivot data for top 20 tickers
pivot_data_top20 = pivot_data.loc[top_15_tickers]

# Create the heatmap using Plotly Graph Objects for more customization
fig = go.Figure(data=go.Heatmap(
    z=pivot_data_top20.values,
    x=pivot_data_top20.columns,
    y=pivot_data_top20.index,
    colorscale='Viridis',
    colorbar=dict(title='Gross'),
    hovertemplate='Ticker: %{y}<br>Year: %{x}<br>Gross: $%{z:.2f}M<extra></extra>'
))

fig.update_layout(
    title='Total Box Office Gross by Ticker and Year (Top 20)',
    xaxis_title='Year',
    yaxis_title='Ticker',
    yaxis_autorange='reversed',
    height=800,
    template='plotly_dark',
    yaxis=dict(tickfont=dict(size=10)),
    xaxis=dict(tickangle=45),
)

fig.show()
