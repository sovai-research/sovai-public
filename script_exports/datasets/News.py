#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# ## News Stream

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


df_news = sov.data("news/daily", tickers=["MSFT","TSLA", "AAPL","NVDA"]); df_news.tail()

# In[4]:


df_news.ticker("NVDA").plot_line("sentiment")

# In[5]:


df_news.ticker("NVDA").plot_line("tone")

# In[6]:


df_news.ticker("NVDA").plot_line("activeness")

# In[7]:


df_news.ticker("NVDA").plot_line("associated_people")

# In[8]:


df_sentiment = sov.data("news/sentiment")

# In[9]:


df_sentiment.sort_values("sentiment")

# In[10]:


df_tone = sov.data("news/tone")

# In[11]:


df_tone.sort_values("tone")

# In[20]:


sov.report("news", report_type="econometric", ticker="AAPL")

# ### Test Trading Strategies

# In[21]:


sov.data("news/daily", tickers=["MSFT"])

# In[29]:


sov.plot("news", chart_type="strategy", ticker='NVDA')

# ### Time Series Topic Modelling

# In[14]:


df_sentiment = sov.data("news/sentiment_score", full_history=True)

# In[15]:


df_sentiment

# In[16]:


df_polarity = sov.data("news/polarity_score", full_history=True)

# In[17]:


df_probability = sov.data("news/topic_probability", full_history=True)

# #### Housing Real Estate (Rolling Median Sentiment Score)

# In[18]:


import plotly.io as pio
# pio.renderers.default = "colab"

df_sentiment.query("calculation == 'sentiment_score_median'")["environmental_sustainability"].rolling(7).mean().plot()

# In[19]:


import plotly.io as pio
# pio.renderers.default = "colab"

df_sentiment.query("calculation == 'sentiment_score_mean'")["environmental_sustainability"].rolling(7).mean().plot()

# #### Interest Rate (Rolling Median Sentiment Score)

# In[20]:


df_sentiment.query("calculation == 'sentiment_score_mean'")["interest_rates"].rolling(7).mean().tail(365).plot()

# In[31]:


sov.plot("news", chart_type="analysis")

# #### Let's train a machine learning model using this data (4 mins)

# In[32]:


df_sentiment.columns

# In[33]:


import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set pandas options to avoid warnings
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

# Fetch S&P 500 data
sp500 = yf.download('^GSPC', start='2016-01-01', end='2024-08-09')
sp500['return'] = sp500['Close'].pct_change()
sp500['forward_return'] = sp500['return'].shift(-22)  # 5 trading days ≈ 1 week
sp500['direction'] = (sp500['forward_return'] > 0).astype(int)

# Function to reshape dataframe
def reshape_dataframe(df):
    df = df.reset_index()
    df_reshaped = df.pivot(index='date', columns='calculation', values=[col for col in df.columns if col not in ['date', 'calculation']])
    df_reshaped.columns = [f'{col[0]}_{col[1]}' for col in df_reshaped.columns]
    return df_reshaped

# Reshape dataframes
df_sentiment_reshaped = reshape_dataframe(df_sentiment)
df_polarity_reshaped = reshape_dataframe(df_polarity)
df_topic_reshaped = reshape_dataframe(df_probability)

# Combine all dataframes
all_dfs = [sp500, df_sentiment_reshaped, df_polarity_reshaped, df_topic_reshaped]
df_combined = pd.concat(all_dfs, axis=1, join='outer')


# Handle missing values
df_combined = df_combined.ffill().bfill()

# df_combined = df_combined[df_combined.index<"2024-08-01"] -> that is crazy it predict august.

# Create rolling features
rolling_windows = [5, 10, 20]
for col in df_combined.columns:
    if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'return', 'forward_return', 'direction']:
        for window in rolling_windows:
            df_combined[f'{col}_rolling_{window}'] = df_combined[col].rolling(window=window).mean()

# Prepare features and target
feature_cols = [col for col in df_combined.columns if col not in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'return', 'forward_return', 'direction']]
X = df_combined[feature_cols].dropna()
y = df_combined['direction'].loc[X.index]

# Scale the features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

# Feature selection
selector = SelectFromModel(lgb.LGBMClassifier(random_state=42), max_features=50)
selector.fit(X_scaled, y)
selected_features = X_scaled.columns[selector.get_support()].tolist()
X_selected = X_scaled[selected_features]

# Set up TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Initialize lists to store performance metrics
metrics = {
    'accuracies': [],
    'precisions': [],
    'recalls': [],
    'f1_scores': []
}

# LightGBM parameters
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_estimators': 100,
    'random_state': 42
}



# Train final model on all data
final_train_data = lgb.Dataset(X_selected, label=y)
final_model = lgb.train(params, final_train_data)



# Make predictions on the entire dataset
df_combined.loc[X_selected.index, 'prob_increase'] = final_model.predict(X_selected)

# Calculate the latest prediction
latest_features = X_selected.iloc[-1].values.reshape(1, -1)
latest_prediction = final_model.predict(latest_features)[0]


# In[40]:


import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "colab"
# Calculate the logarithm of the S&P 500 price
sp500['log_price'] = np.log(sp500['Close'])

# Calculate the 7-day rolling average for the predicted probability
df_combined['prob_increase_rolling'] = df_combined['prob_increase'].rolling(window=30).mean()

# Align the indices of sp500 and df_combined
df_combined = df_combined.reindex(sp500.index)

# Prepare the data for plotting
df_plot = df_combined[["prob_increase_rolling"]].copy()
df_plot['log_price'] = sp500['log_price']

# Create a figure
fig = go.Figure()

# Add the primary y-axis plot for the logarithm of the S&P 500 price
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['log_price'], mode='lines', name='Logarithm of S&P 500 Price', yaxis='y2'))

# Add the secondary y-axis plot for the 7-day rolling average of predicted probability
fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['prob_increase_rolling'], mode='lines', name='7-Day Rolling Probability'))

# Add a horizontal line at 0.5 on the secondary y-axis
fig.add_trace(go.Scatter(
    x=df_plot.index, 
    y=[0.5]*len(df_plot), 
    mode='lines', 
    name='0.5 Threshold', 
    line=dict(dash='dash', color='red'),
))

# Set up the layout to include a secondary y-axis and other essential elements
fig.update_layout(
    yaxis=dict(
        title='7-Day Rolling Probability'
    ),
    yaxis2=dict(
        title='Logarithm of S&P 500 Price',
        overlaying='y',
        side='right',
        # range=[0, 1]  # Set range for probability axis
    ),
    xaxis=dict(
        title='Index'
    ),
    title='Logarithm of S&P 500 Price and 7-Day Rolling Probability with 0.5 Threshold',
    legend=dict(
        orientation="h",  # Horizontal legend
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    hovermode="x unified"  # Unified hover mode for easier comparison
)

# Show the plot
fig.show()


# In[41]:


df_polarity_percentile

# #### Percentile Creations

# In[ ]:


import pandas as pd

# Select the numeric columns (excluding the calculation column)
numeric_columns = df_polarity.select_dtypes(include='number').columns

df_polarity_percentile = df_polarity.copy()
# Calculate the percentile for the numeric columns
df_polarity_percentile[numeric_columns] = df_polarity_percentile.groupby("calculation")[numeric_columns].rank(axis=0,pct=True).rank(axis=1,pct=True) * 100
 
df_polarity_percentile

# ### Another Exploration

# In[51]:


import pandas as pd

df_news = sov.data("news/daily", tickers=["TSLA"]); df_news.tail()

# Reset index to manipulate the datetime
df_news = df_news.reset_index()

# Convert 'datetime' to datetime and extract date without time
df_news['date'] = pd.to_datetime(df_news['date']).dt.date

# Set the MultiIndex back with 'ticker' and 'date'
df_news = df_news.set_index(['ticker', 'date'])


# In[52]:


# Function to resample per ticker
def resample_price(df):
    # Ensure the index is sorted
    df = df.sort_index(level='date')
    
    # Determine the full date range per ticker
    all_dates = pd.date_range(start=df.index.get_level_values('date').min(),
                              end=df.index.get_level_values('date').max(),
                              freq='D')
    
    # Create a MultiIndex with all dates for the ticker
    ticker = df.index.get_level_values('ticker')[0]
    full_index = pd.MultiIndex.from_product([[ticker], all_dates],
                                           names=['ticker', 'date'])
    
    # Reindex the dataframe to include all dates
    df_full = df.reindex(full_index)
    
    # Forward fill missing 'closeadj' values
    df_full = df_full.ffill()
    
    return df_full

df_price = sov.data("market/closeadj",tickers=["TSLA"])

# Apply the resampling function to each ticker
df_price = df_price.groupby(level='ticker').apply(resample_price).reset_index(level=0, drop=True)

# In[53]:


# Apply the resampling function to each ticker
df_news = df_news.groupby(level='ticker').apply(resample_price).reset_index(level=0, drop=True)

# In[54]:


df_price = df_price.tail(90)

# In[55]:


df_price = df_price.droplevel(0)

# In[56]:


df_news = df_news.query("ticker =='TSLA'")

# In[57]:


df_news = df_news.droplevel(0)

# In[58]:


df_news = df_news.rolling(14).mean()

# In[59]:


df_price = df_price.merge(df_news, left_index=True, right_index=True,how="left")

# In[60]:


df_price = df_price[["closeadj","sentiment","polarity"]]

# In[66]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Normalize function
def normalize_to_price(series, price_series):
    price_min = price_series.min()
    price_max = price_series.max()
    series_normalized = ((series - series.min()) / (series.max() - series.min())) * (price_max - price_min) + price_min
    return series_normalized

# Normalize sentiment to price range
sentiment_norm = normalize_to_price(df_price['sentiment'], df_price['closeadj'])

# Create the figure
fig = go.Figure()

# Add sentiment line (will serve as outline)
fig.add_trace(go.Scatter(
    x=df_price.index,
    y=sentiment_norm,
    name='Sentiment',
    line=dict(color='gray', width=1),
    showlegend=False
))

# Add stock price line
fig.add_trace(go.Scatter(
    x=df_price.index,
    y=df_price['closeadj'],
    name='Stock Price',
    line=dict(color='black', width=2)
))

# Add blue sentiment overlay (above price)
fig.add_trace(go.Scatter(
    x=df_price.index,
    y=np.maximum(sentiment_norm, df_price['closeadj']),
    name='Sentiment Above',
    fill='tonexty',
    line=dict(color='blue', width=1),
    fillcolor='rgba(0,0,255,0.2)',
    hovertemplate='Date: %{x}<br>Sentiment: %{text:.3f}<extra></extra>',
    text=df_price['sentiment']
))

# Add red sentiment overlay (below price)
fig.add_trace(go.Scatter(
    x=df_price.index,
    y=np.minimum(sentiment_norm, df_price['closeadj']),
    name='Sentiment Below',
    fill='tonexty',
    line=dict(color='red', width=1),
    fillcolor='rgba(255,0,0,0.2)',
    hovertemplate='Date: %{x}<br>Sentiment: %{text:.3f}<extra></extra>',
    text=df_price['sentiment']
))

# Update layout with improved styling
fig.update_layout(
    title={
        'text': 'Stock Price with Sentiment Overlay',
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=24)
    },
    xaxis_title='Date',
    yaxis_title='Price',
    showlegend=True,
    hovermode='x unified',
    template='plotly_white',
    legend={
        'yanchor':"top",
        'y':0.99,
        'xanchor':"left",
        'x':0.01,
        'bgcolor': 'rgba(255, 255, 255, 0.8)',
        'bordercolor': 'rgba(0, 0, 0, 0.3)',
        'borderwidth': 1
    },
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.1)'
    )
)

# Show the figure
fig.show()

# In[31]:


df_price

# In[21]:


df_news[["sentiment","polarity"]].groupby("ticker").rolling(7).mean()

# In[19]:


df_news.tail()

# In[17]:


df_news

# In[15]:


df_price
