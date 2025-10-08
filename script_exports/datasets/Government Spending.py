#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## Government Spending

# In[20]:


import sovai as sov
sov.token_auth(token="visit https://sov.ai/profile for your token")

# #### Processed Dataset

# In[24]:


df_awards = sov.data("spending/awards",  purge_cache=True, start_date="2025-09-01")

# In[25]:


df_contracts = sov.data("spending/contracts",  purge_cache=True, start_date="2025-09-01")

# In[26]:


df_contracts

# In[5]:


df_contracts = sov.data("spending/contracts", tickers=["AAPL","MSFT","PFE"], start_date="2017-11-20", purge_cache=True)

# In[6]:


df_contracts

# In[10]:


df_contracts

# In[13]:


df_transactions = sov.data("spending/transactions", tickers=["MSFT","TSLA"], purge_cache=True)

# In[14]:


df_transactions

# In[15]:


df_product = sov.data("spending/product", tickers=["MSFT","TSLA"])


# In[16]:


df_product.head()

# In[17]:


df_entities = sov.data("spending/entities", tickers=["MSFT","TSLA"])

# In[18]:


df_entities.tail()

# In[19]:


df_location = sov.data("spending/location", tickers=["MSFT","TSLA"])

# In[20]:


df_location.tail()

# In[21]:


df_competition = sov.data("spending/competition", tickers=["MSFT","TSLA"])


# In[22]:


df_competition.head()

# In[29]:


df_compensation = sov.data("spending/compensation", full_history=True)

# In[31]:


df_compensation.tail()

# In[33]:


df_contracts

# In[44]:


import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_interactive_plots(df_filtered):
    # Create a dropdown widget for ticker selection, defaulting to TSLA
    ticker_dropdown = widgets.Dropdown(
        options=df_filtered['ticker'].unique(),
        value='TSLA' if 'TSLA' in df_filtered['ticker'].unique() else df_filtered['ticker'].unique()[0],
        description='Ticker:',
    )

    # Create a dropdown widget for plot selection
    plot_dropdown = widgets.Dropdown(
        options=[
            'Cumulative Potential Total Value',
            'Dynamic Rolling Average Contract Size'
        ],
        value='Cumulative Potential Total Value',
        description='Select Plot:',
    )

    # Create an output widget for displaying the plot
    output = widgets.Output()

    # Function to update and return the selected plot
    def update_plot(ticker, plot_type):
        # Filter for the selected ticker
        df_ticker = df_filtered[df_filtered['ticker'] == ticker].copy()
        # Ensure 'date' is a datetime
        df_ticker['date'] = pd.to_datetime(df_ticker['date'])

        if plot_type == 'Cumulative Potential Total Value':
            # Resample end-of-month, sum, cumsum, then reset index so Plotly sees the dates
            df_monthly = (
                df_ticker
                .set_index('date')
                .resample('M')['potential_total_value_of_award']
                .sum()
                .cumsum()
                .reset_index()
            )
            fig = px.line(
                df_monthly,
                x='date',
                y='potential_total_value_of_award',
                title='Cumulative Potential Total Value of Award',
                labels={
                    'date': 'Date',
                    'potential_total_value_of_award': 'Cumulative Potential Total Value'
                },
                markers=True,
            )

        else:  # Dynamic Rolling Average
            # dynamic window: 5% of total points (at least 1)
            window = max(int(len(df_ticker) * 0.05), 1)
            df_ticker['rolling_avg'] = (
                df_ticker['potential_total_value_of_award']
                .rolling(window=window)
                .mean()
            )
            fig = px.line(
                df_ticker,
                x='date',
                y='rolling_avg',
                title='Dynamic Rolling Average Contract Size',
                labels={
                    'date': 'Date',
                    'rolling_avg': 'Rolling Average Contract Size'
                },
                markers=True,
            )

        # Common layout tweaks
        fig.update_traces(showlegend=False)
        fig.update_layout(template='plotly_dark', height=400)
        fig.update_xaxes(title_text='Date')
        return fig

    # Event handler to clear and redraw
    def _on_change(change):
        with output:
            clear_output(wait=True)
            fig = update_plot(ticker_dropdown.value, plot_dropdown.value)
            fig.show()

    # Wire up observers
    ticker_dropdown.observe(_on_change, names='value')
    plot_dropdown.observe(_on_change, names='value')

    # Display controls and initial plot
    display(widgets.HBox([ticker_dropdown, plot_dropdown]), output)
    _on_change(None)


# Example call (assuming you have df_contracts defined):
create_interactive_plots(df_contracts.reset_index())

