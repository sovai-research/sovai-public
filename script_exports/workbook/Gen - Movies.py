#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[1]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[2]:


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[16]:


df_movies = sov.data("movies/boxoffice")

# In[17]:


# Reset index to convert 'ticker' and 'date' from MultiIndex to columns

# Ensure 'date' is in datetime format
df_movies['date'] = pd.to_datetime(df_movies['date'])


# Keep only rows where 'date' is within [start_date, max_date]
df_movies = df_movies[df_movies['date'] >= (df_movies["date"].max() - pd.Timedelta(weeks=1))]



# In[18]:


df_movies = df_movies[df_movies["ticker"]!="Private"]

# In[22]:


df_movies

# In[27]:


import pandas as pd

# Step 1: Filter out entries where the ticker is "Private"
df_movies = df_movies[df_movies["ticker"] != "Private"]

# Step 2: Ensure 'date' and 'release_date' are datetime objects
df_movies['date'] = pd.to_datetime(df_movies['date'])
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

# Optional: Handle missing data if necessary
# df_movies = df_movies.dropna(subset=['title', 'date', 'gross', 'theaters', 'days_in_release'])

# Step 3: Create a 'week' column representing the start of each week (Monday)
# Calculate the number of days to subtract to get to Monday
df_movies['week'] = df_movies['date'] - pd.to_timedelta(df_movies['date'].dt.weekday, unit='d')

# Verify the 'week' column
print("Sample 'week' column:")
print(df_movies[['date', 'week']].head())

# Step 4: Verify if each 'title' is associated with a single 'ticker'
title_ticker_counts = df_movies.groupby('title')['ticker'].nunique()
duplicate_titles = title_ticker_counts[title_ticker_counts > 1]

if not duplicate_titles.empty:
    print("\nWarning: The following titles are associated with multiple tickers:")
    print(duplicate_titles)
    # Group by both 'title' and 'ticker' to handle duplicates
    group_by_columns = ['title', 'ticker', 'week']
else:
    group_by_columns = ['title', 'week']

# Step 5: Group by 'title' (and 'ticker' if necessary) and 'week', then aggregate key statistics
weekly_stats = (
    df_movies
    .groupby(group_by_columns)
    .agg(
        total_gross=('gross', 'sum'),
        avg_gross=('gross', 'mean'),
        total_theaters=('theaters', 'sum'),
        avg_theaters=('theaters', 'mean'),
        total_days_in_release=('days_in_release', 'sum'),
        avg_percent_yd=('percent_yd', 'mean'),
        avg_percent_lw=('percent_lw', 'mean')
    )
    .reset_index()
    .sort_values(['title'] + (['ticker'] if 'ticker' in group_by_columns else []) + ['week'])
)

# Step 6: Format numbers for better readability
weekly_stats['total_gross'] = weekly_stats['total_gross'].map('${:,.2f}'.format)
weekly_stats['avg_gross'] = weekly_stats['avg_gross'].map('${:,.2f}'.format)
weekly_stats['total_theaters'] = weekly_stats['total_theaters'].astype(int)
weekly_stats['avg_theaters'] = weekly_stats['avg_theaters'].round(1)
weekly_stats['total_days_in_release'] = weekly_stats['total_days_in_release'].astype(int)
weekly_stats['avg_percent_yd'] = (weekly_stats['avg_percent_yd'] * 100).round(2).astype(str) + '%'
weekly_stats['avg_percent_lw'] = (weekly_stats['avg_percent_lw'] * 100).round(2).astype(str) + '%'

# Optional: Reorder columns for better presentation
columns_order = ['title']
if 'ticker' in group_by_columns:
    columns_order.append('ticker')
columns_order += ['week', 'total_gross', 'avg_gross', 'total_theaters',
                 'avg_theaters', 'total_days_in_release', 'avg_percent_yd', 'avg_percent_lw']

weekly_stats = weekly_stats[columns_order]



# In[29]:


group_by_columns

# In[28]:


weekly_stats

# In[20]:


df_movies["ticker"].value_counts()

# In[7]:


df_movies

# In[5]:


df_movies.sort_values("date")

# In[93]:


df_wiki = sov.data("wikipedia/views", full_history=True)

# In[ ]:



# ================================
# Step 1: Data Preparation
# ================================

# Reset index to convert 'ticker' and 'date' from MultiIndex to columns
df_wiki = df_wiki.reset_index()

# Ensure 'date' is in datetime format
df_wiki['date'] = pd.to_datetime(df_wiki['date'])


# Keep only rows where 'date' is within [start_date, max_date]
df_wiki = df_wiki[df_wiki['date'] >= (df_wiki["date"].max() - pd.Timedelta(weeks=12))]



# Sort the DataFrame by 'ticker' and 'date' to maintain chronological order
df_daily = df_wiki.sort_values(['ticker', 'date']).copy()

# Calculate 'previous_day_search_pressure' by shifting 'search_pressure' by one day within each 'ticker'
df_daily['prev_day'] = df_daily.groupby('ticker')['search_pressure'].shift(1)

# Calculate 'daily_change' as the difference between current and previous search pressures
df_daily['day_chg'] = df_daily['search_pressure'] - df_daily['prev_day']

# Calculate 'average_search_quarter' as the average 'search_pressure' over the last 12 weeks per 'ticker'
df_daily['avg_day'] = df_daily.groupby('ticker')['search_pressure'].transform('mean')

# Calculate 'long_term_change' as the difference between current 'search_pressure' and 'average_search_quarter'
df_daily['ma_day'] = df_daily['search_pressure'] - df_daily['avg_day']

# Drop rows with NaN values (e.g., first day per ticker where 'previous_search_pressure' is NaN)
# Filter to include only rows where 'date' == 'max_date_per_ticker' per ticker
df_daily = df_daily[df_daily['date'] == df_wiki["date"].max() ]

df_daily = df_daily.rename(columns={"search_pressure":"day_search"})

# Select relevant columns
df_daily = df_daily[['ticker', 'date', 'day_search',  
                     'day_chg', 'ma_day']]



import pandas as pd

df_wiki = df_wiki.set_index(["ticker","date"])

# Step 1: Find the global maximum date
max_date = df_wiki.index.get_level_values('date').max()

# Step 2: Get the weekday abbreviation (e.g., 'TUE' for Tuesday)
weekday_abbr = max_date.strftime('%a').upper()[:3]

# Define the resampling frequency to end on the max_date's weekday
resample_freq = f'W-{weekday_abbr}'

df_wiki = (
    df_wiki
    .groupby('ticker')
    .resample(resample_freq, level='date')['search_pressure']
    .mean()
    .reset_index()
)


# Step 1: Sort the DataFrame by 'ticker' and 'date'
df_wiki = df_wiki.sort_values(['ticker', 'date'])


# Step 3: Create 'search_past' by shifting 'search_pressure' by one week within each 'ticker'
df_wiki['search_past'] = df_wiki.groupby('ticker')['search_pressure'].shift(1)

# Step 4: Calculate 'search_change' as the difference between current and past search pressures
df_wiki['search_change'] = df_wiki['search_pressure'] - df_wiki['search_past']

# Optional: Handle NaN values if desired
# df_wiki['search_past'] = df_wiki['search_past'].fillna(0)
# df_wiki['search_change'] = df_wiki['search_change'].fillna(0)

import pandas as pd

# Assuming df_wiki is already processed up to the previous steps

# Step 1: Calculate 'search_quarter' as the average 'search_pressure' over the 12 weeks per 'ticker'
df_wiki['search_quarter'] = df_wiki.groupby('ticker')['search_pressure'].transform('mean')

# Step 2: Calculate 'lt_change' as the difference between 'search_pressure' and 'search_quarter'
df_wiki['lt_change'] = df_wiki['search_pressure'] - df_wiki['search_quarter']

# Optional: If you prefer to round the results for better readability
# df_wiki['search_quarter'] = df_wiki['search_quarter'].round(3)
# df_wiki['lt_change'] = df_wiki['lt_change'].round(3)

# Display the updated DataFrame
print(df_wiki.head())


df_wiki = df_wiki.dropna()



# Filter out rows where 'date' is the max date for each 'ticker'
df_wiki = df_wiki[df_wiki['date'] == df_wiki["date"].max()]


df_wiki.sort_values("search_change")

df_wiki = df_wiki.drop(columns=["search_quarter"])

df_wiki = df_wiki.rename(columns={"search_pressure":"search", "search_past":"last","search_change":"change","lt_change":"ltavg"})

df_wiki = pd.merge(df_wiki.set_index(["ticker","date"]), df_daily.set_index(["ticker","date"]),left_index=True, right_index=True)

df_wiki.head()

import pandas as pd

# Rename columns
df_wiki = df_wiki.rename(columns={
    'day_chg': 'day change',
    'change': 'week change',
    'ma_day': 'short',
    'ltavg': 'long'
})

# Reorder columns with day metrics first, then week metrics
new_order = ['search', 'last', 'week change', 'long', 'day change', 'short']

df_wiki = df_wiki[new_order]

# In[44]:


df_org = sov.data("wikipedia/views", full_history=True)

# In[106]:


import pandas as pd
import numpy as np

def create_pressure_history_columns(df_org, df_wiki, lookback=60):
    """
    Creates pressure history columns from df_org and merges them with df_wiki.

    Parameters:
    - df_org (pd.DataFrame): Original DataFrame with a MultiIndex including 'date'.
    - df_wiki (pd.DataFrame): DataFrame to merge the pressure columns into.
    - lookback (int): Number of days to look back for pressure data.

    Returns:
    - pd.DataFrame: Merged DataFrame with pressure history columns.
    """
    # 1. Get the maximum date from index level 'date'
    max_date = df_org.index.get_level_values('date').max()
    
    # 2. Calculate the cutoff date
    cutoff_date = max_date - pd.Timedelta(days=lookback)
    
    # 3. Filter df_org to only include data after cutoff date
    df_filtered = df_org[df_org.index.get_level_values('date') >= cutoff_date]
    
    # 4. Create pivot table with filtered data
    df_pivot = df_filtered.reset_index().pivot(
        index='ticker', 
        columns='date', 
        values='search_pressure'
    )
    
    # 5. Sort the columns by date ascendingly to ensure pressure_0 is the earliest
    df_pivot = df_pivot.sort_index(axis=1)
    
    # 6. Create pressure column names in ascending order
    num_cols = len(df_pivot.columns)
    all_pressure_cols = [f'pressure_{i}' for i in range(num_cols)]
    df_pivot.columns = all_pressure_cols
    
    # 7. Select columns: first, every 4th, and last
    pressure_cols_to_keep = [all_pressure_cols[0]]  # First column (pressure_0)
    if len(all_pressure_cols) > 2:  # If we have middle columns
        pressure_cols_to_keep.extend(all_pressure_cols[1:-1:4])  # Every 4th column
    pressure_cols_to_keep.append(all_pressure_cols[-1])  # Last column (pressure_{num_cols-1})
    
    # 8. Keep only selected columns
    df_pivot = df_pivot[pressure_cols_to_keep]
    
    # 9. Reset index to make ticker a column
    df_pivot = df_pivot.reset_index()
    
    # 10. Merge with original df_wiki
    df_wiki_expanded = df_wiki.merge(df_pivot, on='ticker', how='left')
    
    return df_wiki_expanded

# Example Usage:

# Assuming you have df_org and df_wiki already defined
# df_org should have a MultiIndex with 'date' and 'ticker'
# df_wiki is the DataFrame you want to expand with pressure columns

# Apply the function
df_wiki_expanded = create_pressure_history_columns(df_org, df_wiki)

# To check the columns we kept and their order
pressure_columns = [col for col in df_wiki_expanded.columns if col.startswith('pressure_')]
print("Pressure columns kept and their order:", pressure_columns)


# In[107]:


df_wiki_expanded[["search","last","week change","long","day change","short"]] = df_wiki_expanded[["search","last","week change","long","day change","short"]]*100

# In[108]:


import pandas as pd


columns_to_process = ['search', 'week change', 'long', 'day change', 'short']

# Initialize an empty list to store DataFrames
top_bottom_dfs = []

# Iterate over each column
for col in columns_to_process:
    # Ensure the column exists in the DataFrame
    if col not in df_wiki_expanded.columns:
        print(f"Column '{col}' does not exist in the DataFrame.")
        continue

    # Sort ascending to get bottom 15
    bottom_15 = df_wiki_expanded.sort_values(by=col, ascending=True).head(15).copy()
    
    # Sort descending to get top 15
    top_15 = df_wiki_expanded.sort_values(by=col, ascending=False).head(15).copy()
    
    # Append to the list
    top_bottom_dfs.extend([top_15, bottom_15])

# Concatenate all DataFrames in the list
combined_df = pd.concat(top_bottom_dfs, ignore_index=True)

# Remove duplicate rows
combined_df_unique = combined_df.drop_duplicates()

# Reset index for cleanliness
combined_df_unique.reset_index(drop=True, inplace=True)


# In[109]:


combined_df_unique = combined_df_unique.sample(100)

# In[110]:


combined_df_unique = combined_df_unique.drop(columns=["long"])

# In[111]:


combined_df_unique = combined_df_unique.rename(columns={"short":"long"})

# In[112]:


combined_df_unique

# In[115]:


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Stock Wikipedia Views Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=combined_df_unique)

# Get pressure column names
pressure_cols = sorted([col for col in combined_df_unique.columns if col.startswith('pressure_')])

# Configure the visualization properties
properties = {
     "visualize": {
        "dark-mode-invert": True,
        "perPage": 20,
        "columns": {
            "ticker": {
                "align": "left",
                "title": "Stock",
                "width": "100"
            },
            "search": {
                "title": "Search",
                "format": "0.000",
                "width": "120"
            },
            "last": {
                "title": "Last",
                "format": "0.000",
                "width": "120"
            },
            "week change": {
                "title": "Change",
                "format": "+0.000",
                "width": 0.27,  # Updated to match working example
                "showAsBar": True,
                "barColorNegative": "#ff4444",
                "fixedWidth": True
            },
            "long": {
                "title": "Long Trend",
                "format": "+0.000",
                "width": "120"
            },
            "day change": {
                "title": "Day Change",
                "format": "+0.000",
                "width": 0.17,  # Updated to match working example
                "fixedWidth": True
            },
            "pressure_0": {
                "type": "number",
                "title": "History",  # Updated to just "History"
                "width": 0.33,  # Updated to match working example
                "format": "0.000",
                "sparkline": {
                    "color": "#18a1cd",
                    "title": "History",
                    "enabled": True,
                    "stroke": 2,
                    "dotMax": True,
                    "dotMin": True,
                    "dotFirst": True,
                    "dotLast": True
                },
                "fixedWidth": True
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

# Configure remaining pressure columns exactly like the first one
for col in pressure_cols[1:]:
    properties["visualize"]["columns"][col] = {
        "type": "number",
        "width": 0.33,  # Updated to match working example
        "format": "0.000",
        "sparkline": {
            "color": "#18a1cd",
            "title": "pressure_history",
            "enabled": True
        },
        "fixedWidth": True
    }

# Set column order
properties["visualize"]["column-order"] = [
    "ticker",
    "search",
    "last",
    "weel change",
    "long",
    "day change",
] + pressure_cols

# Add other visualization settings
properties["describe"] = {
    "intro": "Analysis of Wikipedia page views for stocks with historical pressure trends over 60 days.",
    "byline": "",
    "source-name": "Wikipedia Views Data",
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

# In[114]:


dw.publish_chart("01Wlv")

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
