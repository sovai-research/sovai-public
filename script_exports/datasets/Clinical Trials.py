#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Clinical Trials

# In[2]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# Prediction Data

# In[3]:


df_apps = sov.data("clinical/predict", tickers=["PFE"], start_date = "2024-01-01", end_date= "2025-02-01",  purge_cache=True)

# Trial Descriptions

# In[ ]:


df_trials = sov.data("clinical/trials", tickers=["PFE","LLY"], start_date="2022", purge_cache=True); df_trials.tail()

# Visual Analysis

# In[25]:


df = sov.data("clinical/predict", tickers=["PFE","LLY"], start_date="2022", purge_cache=True)

# In[41]:


import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.express as px
import plotly.graph_objects as go

# ------------- CONFIG -------------
metric   = 'success_prediction'
tickers  = ['LLY', 'PFE']          # order matters: LLY first
colors   = {'LLY': '#1f77b4', 'PFE': '#ff7f0e'}

# ------------- DATA & LOWESS -------------
df = df_apps.reset_index().sort_values(['ticker', 'date'])

def add_loess(g, y=metric, frac=0.2):
    x = (g['date'] - pd.Timestamp('1970-01-01')).dt.days
    g['loess'] = lowess(g[y], x, frac=frac)[:, 1]
    return g

df = df.groupby('ticker', group_keys=False).apply(add_loess)

# ------------- FIGURE 1: shared y‑axis -----------------
fig_shared = px.scatter(
    df, x='date', y=metric, color='ticker',
    template='plotly_dark', opacity=0.35,
    color_discrete_map=colors,
    hover_data={'ticker': False, metric: ':.3f'}
)

# overlay LOWESS lines
for t in tickers:
    sub = df[df['ticker'] == t]
    fig_shared.add_scatter(
        x=sub['date'], y=sub['loess'],
        mode='lines', name=f'{t} trend',
        line=dict(color=colors[t], width=3)
    )

fig_shared.update_layout(
    title=f'{metric.replace("_", " ").title()} – Shared Y‑axis',
    xaxis_title='Date', yaxis_title=metric.replace('_', ' ').title(),
    legend_orientation='h', legend_y=1.05, legend_x=1, legend_xanchor='right',
    margin=dict(t=80, b=40, l=60, r=40), height=450
).update_xaxes(rangeslider_visible=False)

fig_shared.show()

# ------------- FIGURE 2: twin y‑axes -------------------
fig_twin = go.Figure()

for i, t in enumerate(tickers):
    g  = df[df['ticker'] == t]
    ax = 'y' if i == 0 else 'y2'

    # raw dots
    fig_twin.add_scatter(
        x=g['date'], y=g[metric], mode='markers',
        marker=dict(color=colors[t], size=6, opacity=0.4),
        name=f'{t} raw', hoverinfo='none', yaxis=ax
    )
    # trend
    fig_twin.add_scatter(
        x=g['date'], y=g['loess'], mode='lines',
        line=dict(color=colors[t], width=3), name=f'{t} trend', yaxis=ax,
        hovertemplate=f"<b>{t} trend</b><br>Date: %{{x|%b %d, %Y}}<br>{metric}: %{{y:.3f}}<extra></extra>"
    )

    # end‑value annotation
    end_val = g['loess'].iloc[-1]
    fig_twin.add_annotation(
        x=g['date'].iloc[-1], y=end_val, yref=ax,
        text=f"{t}: {end_val:.2f}", showarrow=False,
        font=dict(color=colors[t]), bgcolor='rgba(0,0,0,0.6)', xanchor='left'
    )

# compute tight ranges
rng = {
    t: (df.loc[df['ticker'] == t, 'loess'].min()*0.99,
        df.loc[df['ticker'] == t, 'loess'].max()*1.01)
    for t in tickers
}

fig_twin.update_layout(
    template='plotly_dark',
    title=f'{metric.replace("_", " ").title()} – Twin Y‑axes',
    xaxis=dict(title='Date',
               showgrid=True, gridcolor='rgba(255,255,255,0.1)'),

    # left axis = LLY
    yaxis=dict(
        title=dict(text=f'LLY {metric}', font=dict(color=colors["LLY"])),
        tickfont=dict(color=colors["LLY"]),
        range=rng['LLY'],
        showgrid=False
    ),

    # right axis = PFE
    yaxis2=dict(
        title=dict(text=f'PFE {metric}', font=dict(color=colors["PFE"])),
        tickfont=dict(color=colors["PFE"]),
        overlaying='y', side='right',
        range=rng['PFE'],
        showgrid=False
    ),

    legend=dict(
        orientation='h', y=1.05, x=1, xanchor='right',
        bgcolor='rgba(30,30,30,0.6)', borderwidth=1
    ),

    margin=dict(t=80, b=40, l=60, r=60),
    height=500
)

fig_twin.update_xaxes(rangeslider_visible=False)
fig_twin.show()



# In[12]:


df_apps = sov.data("clinical/trials", start_date="2025-01-20", verbose=False)

# In[15]:


df_apps.sample(5)

# In[42]:


df_apps["links"] = "https://clinicaltrials.gov/study/" + df_apps["trial_id"]

# In[43]:


df_apps.head()

# Trial Information

# In[54]:


df_lly = sov.data("clinical/trials", tickers=["LLY"]) 

# In[55]:


df_lly
