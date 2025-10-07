#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[2]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[3]:


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[4]:


df_short = sov.data("short/price_improvement")

# In[1]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[22]:


df_short_vol_hist = sov.data("short/volume", full_history=True)

# In[26]:


df_short_vol_hist["short_ratio"] = df_short_vol_hist["short_volume"]/df_short_vol_hist["total_volume"]

# In[27]:


df_short_vol_hist.query("ticker == 'AAPL'")

# In[24]:


df_short_vol_hist.tail()

# In[121]:


df_short_vol_levels = sov.data("short/volume")

df_short_vol_previous = sov.data("short/volume", frequency="previous")

df_short_vol_previous["short_ratio"] = df_short_vol_previous["short_volume"]/df_short_vol_previous["total_volume"]

df_short_vol_levels["short_ratio"] = df_short_vol_levels["short_volume"]/df_short_vol_levels["total_volume"]

# Select relevant columns from the previous DataFrame and rename short_ratio
df_prev = df_short_vol_previous[['ticker', 'short_ratio']].rename(columns={
    'short_ratio': 'sr_prev'  # short_ratio from previous date
})

# Select relevant columns from the current DataFrame and rename short_ratio
df_short_vol_levels = df_short_vol_levels.rename(columns={
    'short_ratio': 'sr_curr'  # short_ratio from current date
})

# Merge on 'ticker' using an inner join to include only tickers present in both DataFrames
df_short_vol_levels = pd.merge(df_short_vol_levels, df_prev, on='ticker', how='inner')


df_short_vol_levels[["short_volume","total_volume"]] = df_short_vol_levels[["short_volume","total_volume"]]/1000

df_short_vol_levels = df_short_vol_levels.rename(columns={"short_volume":"short_vol","total_volume":"tot_vol","short_volume_ratio_exchange":"sr_ex","retail_short_ratio":"sr_ret","institutional_short_ratio":"sr_int","market_maker_short_ratio":"sr_mm"})

df_short_vol_levels = df_short_vol_levels.dropna()

df_short_vol_levels = df_short_vol_levels.drop(columns=["first_two_letters"])

df_short_vol_levels["sr_diff"] = df_short_vol_levels["sr_curr"] - df_short_vol_levels["sr_prev"] 

df_short_vol_levels = df_short_vol_levels[["ticker","date","short_vol","tot_vol","sr_curr","sr_prev","sr_diff","sr_ex","sr_ret","sr_int","sr_mm"]]



# In[122]:


df_short_vol_levels[["sr_curr","sr_prev","sr_diff","sr_ex","sr_ret","sr_int","sr_mm"]] = df_short_vol_levels[["sr_curr","sr_prev","sr_diff","sr_ex","sr_ret","sr_int","sr_mm"]]*100
df_short_vol_levels


# In[123]:


import sovai as sov
import pandas as pd


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[124]:


df_short_vol_levels = df_short_vol_levels.set_index(["ticker","date"])

df_short_vol_levels = df_short_vol_levels.filter(["market_cap>100"])

# In[125]:


# Sort by 'sr_diff' and get top 30 and bottom 30, concatenate them
df_sorted = df_short_vol_levels.sort_values("sr_diff").reset_index()
df_top_bottom = pd.concat([df_sorted.head(30), df_sorted.tail(30)], ignore_index=True)

# In[126]:


max_date = df_top_bottom["date"].max()
formatted_date = max_date.strftime('%B %d, %Y')
df_top_bottom = df_top_bottom.drop(columns=["date"])

# In[ ]:


df_top_bottom = df_top_bottom.drop(columns=["sr_ret","sr_int","sr_mm"])

# In[144]:


df_top_bottom = df_top_bottom.rename(columns={"sr_curr":"short_ratio"})

# In[145]:


from datawrapper import Datawrapper

# Initialize Datawrapper with the access token
dw = Datawrapper(access_token="your_token")

# Create a new chart for short volume changes
chart2 = dw.create_chart(
    title="Short Volume Ratio Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart2['id'], data=df_top_bottom)

# Configure the visualization properties
metadata = {
    "visualize": {
        "columns": {
            "ticker": {
                "title": "Ticker", 
                "align": "left",
                "width": "100"
            },
            "sr_diff": {
                "title": "SR Change", 
                "format": "0.0%",
                "showAsBar": True,
                "width": 0.61,
                "fixedWidth": True,
                "style": {"color": False},
                "barColor": 7,
                "barColorNegative": 4
            },
            "short_vol": {
                "title": "Short Volume (K)", 
                "format": "0,0.0",
                "width": "120",
                "borderLeft": "none"
            },
            "tot_vol": {
                "title": "Total Volume (K)", 
                "format": "0,0.0",
                "width": "120",
                "borderLeft": "none"
            },
            "short_ratio": {
                "title": "Current SR", 
                "format": "0.0%",
                "width": "100",
                "style": {"color": "#c71e1d", "background": False},
                "borderLeft": "none"
            },
            "sr_prev": {
                "title": "Previous SR", 
                "format": "0.0%",
                "width": "100",
                "style": {"color": "#09bb9f"},
                "borderLeft": "none"
            },
            "sr_ex": {
                "title": "SR Expected", 
                "format": "0.0%",
                "width": "100",
                "borderLeft": "none"
            }
        },
        "dark-mode-invert": True,
        "sortBy": "sr_diff",
        "sortDirection": "asc",
        "perPage": 15,
        "pagination": {
            "enabled": True,
            "position": "bottom",
            "rowsPerPage": 15
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 1.1,
                "background": False,
                "color": False,
                "italic": False
            },
            "borderTop": "none",
            "borderBottom": "2px",
            "borderTopColor": "#333333",
            "borderBottomColor": "#333333"
        },
        "showHeader": True,
        "markdown": False,
        "striped": False,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": False,
        "mergeEmptyCells": False,

    },
    "describe": {
        "intro": "Analysis of largest short ratio changes. The table shows short volume, total volume, and short ratio (SR) metrics including current SR, previous SR, and the change between them.",
        "byline": "",
        "source-name": "Market Data",
        "source-url": "",
        "hide-title": False
    },
    "publish": {
        "embed-width": 648,
        "embed-height": 776,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        }
    }
}
# Update the chart with the metadata
dw.update_chart(
    chart_id=chart2['id'],
    metadata=metadata
)

# Publish the chart
dw.publish_chart(chart2['id']) 

# Get the URL of the published chart
chart_urls = dw.get_chart_display_urls(chart2['id'])
print("\nChart URLs:", chart_urls)

# In[131]:


df_top_bottom

# In[111]:


df_short_vol_levels.sort_values("sr_diff").head(50)

# In[ ]:




def get_bankruptcy_date(frequency="difference"):

    df = sov.data("bankruptcy", frequency=frequency)
    
    df = df.set_index(["ticker","date"])
    
    df = df.filter(["market_cap>100"])
    
    df[["sans_market","probability"]].reset_index()
    
    df = pd.merge(df[["sans_market","probability"]].reset_index(),tickers_meta[["ticker","sector"]], on="ticker", how="left")
    
    df = df.rename(columns={"sans_market":"sansmarket"})
    return df


# In[94]:


df_short_vol_levels.sample(20)

# In[17]:


df_short_vol_levels["ratio"] =  df_short_vol_levels["short_volume"]  / df_short_vol_levels["total_volume"]

# In[19]:


df_short_vol_levels["short_disparity"]  = df_short_vol_levels["short_volume_ratio_exchange"] -  df_short_vol_levels["ratio"]

# In[21]:


df_short_vol_levels.sort_values("short_disparity")

# In[14]:


df_short_vol_change

# In[9]:


df_short_volume.dropna().sort_values("short_volume_ratio_exchange")

# In[7]:


df_short_volume

# In[62]:


df_short_volume = sov.data("short/volume")

# In[61]:


df_short_volume.head()

# In[140]:


dw.publish_chart("RUqwn")

# In[22]:


df_shorted.head()

# In[3]:


### Here make space for some utility functions.
import os
from dotenv import load_dotenv
from github import Github
import plotly.graph_objects as go

def upload_file_to_github(plot_html, file_path_in_repo, commit_message="Add/update plot",
                         env_path='/Users/dereksnow/Sovai/GitHub/SovAI/.env', repo_name="sovai-research/sovai-research.github.io", git_token_env_var='GIT_TOKEN'):
    """
    Uploads or updates a file in the specified GitHub repository.

    Parameters:
    - plot_html (str): The content of the file to upload (e.g., HTML string).
    - file_path_in_repo (str): The path in the repository where the file will be uploaded (e.g., 'plots/risks/turing_risk_plot.html').
    - commit_message (str): The commit message. Defaults to "Add/update plot".
    - env_path (str, optional): The path to the .env file. If None, load_dotenv() will search automatically.
    - repo_name (str, optional): The full name of the repository (e.g., "owner/repo"). If None, read from environment variable 'GIT_REPO_NAME'.
    - git_token_env_var (str, optional): The name of the environment variable that holds the GitHub token. Defaults to 'GIT_TOKEN'.

    Returns:
    - None

    Raises:
    - ValueError: If the GitHub token or repository name is not provided.
    - Exception: For other unexpected errors during the upload/update process.
    """

    # Load environment variables
    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()  # Automatically load from .env in current or parent directories

    # Retrieve GitHub token from environment variables
    GIT_TOKEN = os.getenv(git_token_env_var)
    if not GIT_TOKEN:
        raise ValueError(f"{git_token_env_var} not found in environment variables. Please set it in the .env file or environment.")

    # Initialize GitHub client
    g = Github(GIT_TOKEN)

    # Retrieve repository name
    if not repo_name:
        repo_name = os.getenv('GIT_REPO_NAME')
        if not repo_name:
            raise ValueError("Repository name not specified and 'GIT_REPO_NAME' not found in environment variables.")

    # Initialize repository
    try:
        repo = g.get_repo(repo_name)
        print(f"Successfully connected to repository: {repo.full_name}")
    except Exception as e:
        print(f"Error accessing repository '{repo_name}': {e}")
        raise e

    # Upload or update the file
    try:
        # Attempt to get the existing file
        contents = repo.get_contents(file_path_in_repo)
        # If the file exists, update it
        repo.update_file(contents.path, commit_message, plot_html, contents.sha)
        print(f"File '{file_path_in_repo}' updated successfully.")
    except Exception as e:
        if "404" in str(e):
            # If the file does not exist, create it
            repo.create_file(file_path_in_repo, commit_message, plot_html)
            print(f"File '{file_path_in_repo}' created successfully.")
        else:
            print(f"An error occurred: {e}")
            raise e


# In[4]:


df_congress = sov.data("congress")

# In[21]:


import pandas as pd
from datetime import timedelta
import requests
import numpy as np

# Convert 'date' and 'transaction_date' to datetime
df_congress['date'] = pd.to_datetime(df_congress['date'])
df_congress['transaction_date'] = pd.to_datetime(df_congress['transaction_date'])

# Find the latest date
latest_date = df_congress['date'].max()
print(f"Latest Date: {latest_date.date()}")

# Calculate start date (7 days before latest_date)
start_date = latest_date - timedelta(days=7)
print(f"Start Date: {start_date.date()}")

# Filter for the last 7 days
df_last_7_days = df_congress[df_congress['date'] >= start_date].copy()
print(f"Number of Transactions in Last 7 Days: {df_last_7_days.shape[0]}")

# Get unique tickers in the last 7 days
unique_tickers = df_last_7_days['ticker'].unique()
print(f"Unique Tickers in Last 7 Days: {unique_tickers}")

# Define base URL for raw GitHub content
base_url = 'https://raw.githubusercontent.com/nvstly/icons/main/ticker_icons/'

# Define default logo URL
default_logo_url = 'https://avatars.githubusercontent.com/u/114351196?s=48&v=4'  # Replace with actual default if different

# Function to assign logo URL with error handling
def get_logo_url(ticker):
    url = f"{base_url}{ticker}.png"
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            return url
        else:
            return default_logo_url
    except requests.RequestException:
        return default_logo_url

# Create a dictionary mapping tickers to their logo URLs
ticker_logo_mapping = {ticker: get_logo_url(ticker) for ticker in unique_tickers}

# Display the mapping
for ticker, logo_url in ticker_logo_mapping.items():
    print(f"Ticker: {ticker}, Logo URL: {logo_url}")

# Assign logo URLs where 'date' >= start_date, else set to None
df_congress['logo'] = np.where(
    df_congress['date'] >= start_date,
    df_congress['ticker'].map(ticker_logo_mapping),
    None  # Or assign a default value like default_logo_url
)

# Alternatively, using pd.Series.where
# ticker_logo_series = pd.Series(ticker_logo_mapping)
# df_congress['logo'] = df_congress['ticker'].map(ticker_logo_series).where(df_congress['date'] >= start_date)

# Display the updated DataFrame
print(df_congress[['ticker', 'date', 'logo']])

# Filter congress dataframe based on date
df_congress = df_congress[df_congress['date'] >= start_date]

from datawrapper import Datawrapper
import pandas as pd

# Function to get price history for a ticker and convert to columns
def get_price_history_columns(ticker):
    try:
        # Get price data using your SDK
        df_price = sov.data("market/closeadj", tickers=[ticker])
        # Get last 90 days of data
        latest_prices = df_price.tail(90)
        # Convert dates to column names and values to a row
        return latest_prices['closeadj'].values
    except:
        return None

# Create a list to store the data
all_data = []

# Process each ticker
for idx, row in df_congress.iterrows():
    prices = get_price_history_columns(row['ticker'])
    if prices is not None:
        data_dict = {
            'ticker': row['ticker'],
            'logo': row['logo'],
            'transaction_date': row['transaction_date'],
            'representative': row['representative'],
            'bio_guide_url': row['bio_guide_url'],
            'transaction': row['transaction'],
            'amount': row['amount'],
            'house': row['house'],
            'transaction': row['transaction'],
            'party': row['party'],
            'days_to_report': row['days_to_report'],
        }
        # Add price columns
        for i, price in enumerate(prices):
            data_dict[f'price_{i}'] = price
        all_data.append(data_dict)

# Create new dataframe with price columns
df_with_prices_org = pd.DataFrame(all_data)


# In[47]:


df_with_prices = df_with_prices_org.copy()


# Format the data to include logos and links

# df_with_prices['ticker'] = df_with_prices.apply(
#     lambda row: f"{row['ticker']}<tooltip>{row['logo']}</tooltip>", 
#     axis=1
# )

df_with_prices['logo'] = df_with_prices['logo'].apply(lambda x: f'![logo]({x})')
df_with_prices['representative'] = df_with_prices.apply(
    lambda x: f'[{x["representative"]}]({x["bio_guide_url"]})', axis=1
)


df_with_prices['house_party'] = df_with_prices.apply(
    lambda x: f"{'Sen' if x['house']=='Senate' else 'Rep'} ({x['party']})", 
    axis=1
)


df_with_prices = df_with_prices.drop(columns=["bio_guide_url","party","house","days_to_report","logo"])

# # Format ticker with logo and Yahoo link
df_with_prices['ticker'] = df_with_prices.apply(
    lambda x: f"[{x['ticker']}](https://finance.yahoo.com/quote/{x['ticker']})", 
    axis=1
)

# # # Format ticker with logo and Yahoo link
# df_with_prices['transaction'] = df_with_prices.apply(
#     lambda x: f"[{x['transaction']}: {x['logo']}", 
#     axis=1
# )



# Get the list of column names
columns = df_with_prices.columns

# Filter for price_ columns
price_columns = [col for col in columns if col.startswith('price_')]

# Always keep the first and last price_ columns, and select every 4th column in between
selected_price_columns = [price_columns[0]] + price_columns[1:-1:4] + [price_columns[-1]]

# Define the final columns to keep, including the non-price columns
non_price_columns = [col for col in columns if not col.startswith('price_')]
final_columns = non_price_columns + selected_price_columns

# Create a new dataframe with the selected columns
df_with_prices = df_with_prices[final_columns]


df_with_prices = df_with_prices[[col for col in df_with_prices.columns if col not in ['house_party', 'transaction', 'amount',]] + ['house_party','transaction', 'amount']]

df_with_prices = df_with_prices.rename(columns={"transaction":"type", "transaction_date":"transaction","house_party":"house"})

# Display the result
df_with_prices.head()


# In[51]:


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Congressional Stock Trades with Price History",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=df_with_prices)

# Get price column names
price_cols = [col for col in df_with_prices.columns if col.startswith('price_')]

# Configure the visualization properties
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 20,
        "columns": {
            "ticker": {
                "align": "left",
                "title": "Stock",
                "tooltip": {
                    "enabled": True,
                    "template": "{{logo}}"
                }
            },
            "transaction": {
                "title": "Date",
                "format": "YYYY-MM-DD"
            },
            "representative": {
                "title": "Representative",
                "width": 0.23,
                "markdown": True,
                "fixedWidth": True
            },
            "price_0": {  
                "type": "number",
                "title": "Price History",
                "width": 0.28,
                "format": "$.2f",
                "visible": True,
                "sparkline": {
                    "color": "#18a1cd",
                    "title": "stock_price",
                    "enabled": True,
                    "stroke": 2,
                    "dotMax": True,
                    "dotMin": True,
                    "dotFirst": True,
                    "dotLast": True
                },
                "fixedWidth": True
            },
            "type": {
                "title": "Type",
                "customColor": True,
                "customColorBy": "type",
                "customColorText": {
                    "Sale": "#ee493a",
                    "Exchange": "#18a1cd",
                    "Purchase": "#09bb9f",
                    "Sale (Partial)": "#ffb239"
                }
            },
            "amount": {
                "title": "Amount",
                "format": "$0,0",
                "align": "right",
                "showAsBar": True,
                "fixedWidth": True
            },
            "house": {
                "title": "Chamber"
            }
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 0.9,
                "color": "#494949"
            },
            "borderBottom": "2px",
            "borderBottomColor": "#333333"
        },
        "pagination": {
            "enabled": True,
            "position": "bottom",
            "pagesPerScreen": 10
        },
        "striped": True,
        "markdown": True,
        "showHeader": True,
        "compactMode": True,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": False,
        "mergeEmptyCells": False
    }
}

# Configure remaining price columns
for col in price_cols[1:]:
    properties["visualize"]["columns"][col] = {
        "type": "number",
        "width": 0.28,
        "format": "$.2f",
        "visible": True,
        "sparkline": {
            "color": "#18a1cd",
            "title": "stock_price",
            "enabled": True
        },
        "fixedWidth": True,
        "showOnMobile": False,
        "showOnDesktop": True
    }

# Set column order
properties["visualize"]["column-order"] = [
    "ticker",
    "transaction",
    "representative",
    "price_0"
] + price_cols[1:] + [
    "type",
    "amount",
    "house"
]

# Add other visualization settings
properties["describe"] = {
    "intro": "Congressional stock trades with 90-day historical price performance.",
    "byline": "",
    "source-name": "Congress Trading Data",
    "source-url": "",
    "hide-title": False
}

properties["publish"] = {
    "embed-width": 682,
    "embed-height": 1086,
    "blocks": {
        "logo": {"enabled": False},
        "embed": False,
        "download-pdf": False,
        "download-svg": False,
        "get-the-data": True,
        "download-image": False
    },
    "autoDarkMode": False,
    "chart-height": 988,
    "force-attribution": False
}

# Update the chart with the properties
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)
