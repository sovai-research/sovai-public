#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[39]:


from datetime import datetime
import requests
import json

# Set up Notion credentials (hardcoded as per your request)
NOTION_TOKEN = "your_notion_token_here"  # **Ensure this token is kept secure!**
DATABASE_ID = "your_database_id_here"
NOTION_VERSION = "2022-06-28"

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": NOTION_VERSION,
}

def create_page(title, database_id, children):
    """
    Creates a new page in the specified Notion database.

    Args:
        title (str): The title of the page.
        database_id (str): The ID of the Notion database.
        children (list): A list of block objects to include in the page.

    Returns:
        dict: The response from the Notion API.
    """
    page_data = {
        "parent": {"database_id": database_id},
        "properties": {
            "Title": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
        },
        "children": children
    }

    response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=page_data)
    return response


def find_page_by_title(database_id, title):
    """
    Searches the Notion database for a page with the specified title.

    Args:
        database_id (str): The ID of the Notion database.
        title (str): The title to search for.

    Returns:
        dict or None: The page object if found, else None.
    """
    query_url = f"https://api.notion.com/v1/databases/{database_id}/query"
    query_data = {
        "filter": {
            "property": "Title",
            "title": {
                "equals": title
            }
        }
    }

    response = requests.post(query_url, headers=headers, json=query_data)
    
    if response.status_code != 200:
        print("Failed to query database:")
        print(json.dumps(response.json(), indent=2))
        return None

    results = response.json().get("results")
    if results:
        return results[0]  # Assuming titles are unique
    return None


def append_to_page(page_id, children):
    """
    Appends new blocks to an existing Notion page.

    Args:
        page_id (str): The ID of the page to append to.
        children (list): A list of block objects to append.

    Returns:
        dict: The response from the Notion API.
    """
    append_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    append_data = {
        "children": children
    }
    response = requests.patch(append_url, headers=headers, json=append_data)
    return response


def build_content_from_dict(content_dict):
    """
    Builds Notion content blocks from a dictionary.

    Args:
        content_dict (dict): A dictionary containing content definitions.

    Returns:
        list: A list of Notion block objects.
    """
    children = []

    # Add Heading
    if "heading" in content_dict and content_dict["heading"]:
        children.append(
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["heading"]
                            }
                        }
                    ]
                },
            }
        )

    # Add Content
    if "content" in content_dict and content_dict["content"]:
        children.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["content"]
                            }
                        }
                    ]
                },
            }
        )

        # Add List Items (Bullet Points)
    if "list" in content_dict and content_dict["list"]:
        list_blocks = build_bullet_list(content_dict["list"])
        children.extend(list_blocks)
        
    # Add URL as a Link
    if "url" in content_dict and content_dict["url"]:
        children.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content_dict["url"],
                                "link": {"url": content_dict["url"]}
                            }
                        }
                    ]
                },
            }
        )



    return children


def build_bullet_list(items):
    """
    Builds Notion bullet list blocks from a list of items.

    Args:
        items (list): A list of strings representing bullet points.

    Returns:
        list: A list of Notion bulleted list item block objects.
    """
    bullet_blocks = []
    for item in items:
        bullet_blocks.append(
            {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": item
                            }
                        }
                    ]
                },
            }
        )
    return bullet_blocks


def build_children_from_sections(content_sections):
    """
    Iterates through the content sections dictionary and builds the children blocks.

    Args:
        content_sections (dict): Dictionary containing all content sections.

    Returns:
        list: A list of Notion block objects.
    """
    children = []
    for key in sorted(content_sections.keys()):
        section = content_sections[key]
        section_blocks = build_content_from_dict(section)
        children.extend(section_blocks)
    return children


def handle_page_creation_or_append(title, database_id, content_sections):
    """
    Handles the logic to either create a new page or append content to an existing page.

    Args:
        title (str): The title of the page.
        database_id (str): The ID of the Notion database.
        content_sections (dict): Dictionary containing all content sections.

    Returns:
        None
    """
    try: 
        current_date = datetime.now().strftime("%Y-%m-%d")
    except:
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    full_title = f"{title} - {current_date}"

    # Build the content blocks
    children = build_children_from_sections(content_sections)

    # Check if the page already exists
    existing_page = find_page_by_title(database_id, full_title)

    if existing_page:
        print(f"Page '{full_title}' already exists. Appending new content to it.")
        page_id = existing_page["id"]
        response = append_to_page(page_id, children)
        
        if response.status_code == 200:
            print("New content appended successfully.")
            # Construct the page URL manually
            # Note: Notion page URLs follow the format https://www.notion.so/{workspace}/{page_id}
            # However, constructing the exact URL might require additional steps.
            # Here, we'll provide a placeholder.
            page_url = f"https://www.notion.so/{page_id.replace('-', '')}"
            print(f"View your page here: {page_url}")
        else:
            print("Failed to append new content:")
            print(json.dumps(response.json(), indent=2))
    else:
        print(f"Page '{full_title}' does not exist. Creating a new page with the new content.")
        response = create_page(full_title, database_id, children)
        
        # Handle the response
        if response.status_code == 200:
            page_url = response.json().get("url", "No URL returned")
            print("Page created successfully with the new content.")
            print(f"View your page here: {page_url}")
        else:
            print("Failed to create page:")
            print(json.dumps(response.json(), indent=2))


# In[1]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[2]:


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[22]:


df_sentiment = sov.data("news/sentiment",full_history=True)

df_sentiment = df_sentiment.filter(["market_cap>50"])

import pandas as pd

# ================================
# Step 1: Data Preparation
# ================================

# Reset index to convert 'ticker' and 'date' from MultiIndex to columns
df_sentiment = df_sentiment.reset_index()

# Ensure 'date' is in datetime format
df_sentiment['date'] = pd.to_datetime(df_sentiment['date'])

# Step 1: Find the global maximum date
max_date = df_sentiment['date'].max()

# Calculate the start date (12 weeks before the max_date)
start_date = max_date - pd.Timedelta(weeks=12)

# Keep only rows where 'date' is within the last 12 weeks from the maximum date
df_sentiment = df_sentiment[df_sentiment['date'] >= start_date]

# Sort the DataFrame by 'ticker' and 'date' to maintain chronological order
df_sentiment = df_sentiment.sort_values(['ticker', 'date']).copy()


# Reset index to convert 'ticker' and 'date' from MultiIndex to columns
df_sentiment_reset = df_sentiment.reset_index()

# Drop 'level_0' column if it exists
if 'level_0' in df_sentiment_reset.columns:
    df_sentiment_reset = df_sentiment_reset.drop(columns=['level_0'])

# Ensure 'date' is in datetime format
df_sentiment_reset['date'] = pd.to_datetime(df_sentiment_reset['date'])

# Pivot the DataFrame: rows -> 'ticker', columns -> 'date', values -> 'sentiment'
df_pivot = df_sentiment_reset.pivot(index='ticker', columns='date', values='sentiment')

# Create a complete date range from the minimum to the maximum date
complete_date_range = pd.date_range(start=df_pivot.columns.min(), end=df_pivot.columns.max(), freq='D')

# Reindex the pivoted DataFrame to include all dates
df_pivot = df_pivot.reindex(columns=complete_date_range)

# Forward-fill missing sentiment values along the date axis
df_pivot = df_pivot.ffill(axis=1)

# Stack the DataFrame back to long format
df_filled = df_pivot.stack().reset_index()

# Rename the columns appropriately
df_filled.columns = ['ticker', 'date', 'sentiment']

# Set the MultiIndex back to ['ticker', 'date'] if needed
df_filled = df_filled.set_index(['ticker', 'date'])

# Display the first few rows of the filled DataFrame
print("DataFrame after forward-filling missing dates:")
print(df_filled.head(15))


# Calculate 6-day rolling average by ticker
df_filled["sentiment"] = df_filled.groupby('ticker')['sentiment'].rolling(window=6).mean().values


df_filled = df_filled.dropna()

# In[23]:


df_filled.query("ticker == 'AAPL'")

# In[42]:



# ================================
# Step 2: Daily Metrics Calculation
# ================================

# Sort the DataFrame by 'ticker' and 'date' to maintain chronological order
df_daily_sentiment = df_filled.reset_index().sort_values(['ticker', 'date']).copy()

# Calculate 'last' sentiment by shifting 'sentiment' by one day within each 'ticker'
df_daily_sentiment['last'] = df_daily_sentiment.groupby('ticker')['sentiment'].shift(1)

# Calculate 'day change' as the difference between current and previous sentiment
df_daily_sentiment['day change'] = df_daily_sentiment['sentiment'] - df_daily_sentiment['last']

# Calculate 'long' as the difference between current 'sentiment' and the average sentiment over the last 12 weeks per 'ticker'
df_daily_sentiment['long'] = df_daily_sentiment['sentiment'] - df_daily_sentiment.groupby('ticker')['sentiment'].transform('mean')

# Drop rows with NaN values (e.g., first day per ticker where 'last' is NaN)
df_daily_sentiment = df_daily_sentiment.dropna()

# Filter to include only rows where 'date' == 'max_date' per ticker
df_daily_sentiment = df_daily_sentiment[df_daily_sentiment["date"].max() == df_daily_sentiment['date']]

# Rename 'sentiment' to 'day_sentiment' for clarity
df_daily_sentiment = df_daily_sentiment.rename(columns={"sentiment": "day_sentiment"})

# Select relevant columns
df_daily_sentiment = df_daily_sentiment[['ticker', 'date', 'day_sentiment', 'day change', 'long']]

# ================================
# Step 3: Weekly Resampling and Metrics Calculation
# ================================

# Set the MultiIndex back for resampling
df_sentiment = df_sentiment.set_index(["ticker", "date"])

# Step 2: Get the weekday abbreviation (e.g., 'TUE' for Tuesday) from the max_date
weekday_abbr = max_date.strftime('%a').upper()[:3]

# Define the resampling frequency to end on the max_date's weekday
resample_freq = f'W-{weekday_abbr}'

# Resample sentiment data weekly, taking the mean sentiment per week
df_sentiment_resampled = (
    df_sentiment
    .groupby('ticker')
    .resample(resample_freq, level='date')['sentiment']
    .mean()
    .reset_index()
)

# Sort the resampled DataFrame by 'ticker' and 'date'
df_sentiment_resampled = df_sentiment_resampled.sort_values(['ticker', 'date']).copy()

# Create 'last' sentiment by shifting 'sentiment' by one week within each 'ticker'
df_sentiment_resampled['last'] = df_sentiment_resampled.groupby('ticker')['sentiment'].shift(1)

# Calculate 'week change' as the difference between current and previous sentiment
df_sentiment_resampled['week change'] = df_sentiment_resampled['sentiment'] - df_sentiment_resampled['last']

# Calculate 'long_term_change' as the difference between current 'sentiment' and the average sentiment over the last 12 weeks per 'ticker'
df_sentiment_resampled['long'] = df_sentiment_resampled['sentiment'] - df_sentiment_resampled.groupby('ticker')['sentiment'].transform('mean')

# Drop rows with NaN values (e.g., first week per ticker where 'last' is NaN)
df_sentiment_resampled = df_sentiment_resampled.dropna()

# Filter to include only rows where 'date' == 'max_date' per ticker
df_sentiment_resampled = df_sentiment_resampled[df_sentiment_resampled["date"].max() == df_sentiment_resampled['date']]

# Rename 'sentiment' to 'week_sentiment' for clarity
df_sentiment_resampled = df_sentiment_resampled.rename(columns={"sentiment": "week_sentiment"})

# Select relevant columns
df_sentiment_resampled = df_sentiment_resampled[['ticker', 'date', 'week_sentiment','last', 'week change', 'long']]

# ================================
# Step 4: Merging Daily and Weekly Metrics
# ================================

# Merge the daily and weekly sentiment DataFrames on ['ticker', 'date']
df_final_sentiment = pd.merge(
    df_sentiment_resampled,
    df_daily_sentiment,
    on=['ticker', 'date'],
    how='inner'
)

# ================================
# Step 5: Final Cleanup and Column Ordering
# ================================



# In[43]:


# Rename columns to match the desired 'new_order'
df_final_sentiment = df_final_sentiment.rename(columns={
    'week_sentiment': 'sentiment',       # Current sentiment from weekly data
    'last': 'last',                    # Last week's sentiment
    'change': 'week change',      # Weekly change
    'long_x': 'long',                    # Long-term change from weekly data
    'day change': 'day change',          # Daily change
    'long_y': 'short'                    # Short-term change from daily data
})

# Select and reorder the columns
new_order = ['ticker', 'sentiment', 'last', 'week change', 'long', 'day change', 'short']

# Ensure all columns in new_order exist in df_final_sentiment
existing_columns = [col for col in new_order if col in df_final_sentiment.columns]
missing_columns = [col for col in new_order if col not in df_final_sentiment.columns]

if missing_columns:
    print(f"Warning: The following columns are missing and will be excluded from the final DataFrame: {missing_columns}")

# Reorder the DataFrame
df_final_sentiment = df_final_sentiment[existing_columns]

# Display the final DataFrame
print(df_final_sentiment.head())

df_final_sentiment = df_final_sentiment.drop(columns=["long"]).rename(columns={"short":"long"})

df_final_sentiment

df_final_sentiment = df_final_sentiment.reset_index(drop=True)

df_final_sentiment = df_final_sentiment.sort_values("week change")

df_final_sentiment

# In[44]:


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
        values='sentiment'
    )
    
    # 5. Sort the columns by date ascendingly to ensure pressure_0 is the earliest
    df_pivot = df_pivot.sort_index(axis=1)
    
    # 6. Create pressure column names in ascending order
    num_cols = len(df_pivot.columns)
    all_pressure_cols = [f'sent_{i}' for i in range(num_cols)]
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


df_sentiment_expanded = create_pressure_history_columns(df_filled, df_final_sentiment)

# To check the columns we kept and their order
pressure_columns = [col for col in df_sentiment_expanded.columns if col.startswith('sent_')]
print("Pressure columns kept and their order:", pressure_columns)


df_sentiment_expanded.query("ticker == 'AAPL'").tail(20)

df_sentiment_expanded[["sentiment","last","week change","long","day change"]] = df_sentiment_expanded[["sentiment","last","week change","long","day change"]]*100

import pandas as pd


columns_to_process = ['sentiment', 'week change', 'long', 'day change']

# Initialize an empty list to store DataFrames
top_bottom_dfs = []

# Iterate over each column
for col in columns_to_process:
    # Ensure the column exists in the DataFrame
    if col not in df_sentiment_expanded.columns:
        print(f"Column '{col}' does not exist in the DataFrame.")
        continue

    # Sort ascending to get bottom 15
    bottom_15 = df_sentiment_expanded.sort_values(by=col, ascending=True).head(50).copy()
    
    # Sort descending to get top 15
    top_15 = df_sentiment_expanded.sort_values(by=col, ascending=False).head(50).copy()
    
    # Append to the list
    top_bottom_dfs.extend([top_15, bottom_15])

# Concatenate all DataFrames in the list
combined_df = pd.concat(top_bottom_dfs, ignore_index=True)

# Remove duplicate rows
combined_df_unique = combined_df.drop_duplicates()

# Reset index for cleanliness
combined_df_unique.reset_index(drop=True, inplace=True)


combined_df_unique = combined_df_unique.dropna()

# In[27]:


import datetime

import locale

# Set locale to US English
locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')


def get_week_ending_label(reference_date=None):
    """
    Returns a formatted string indicating the week ending on the last Friday relative to the reference date.

    Args:
        reference_date (datetime.date, optional): The date to reference. Defaults to today.

    Returns:
        str: Formatted string like "Week ending Friday 25th October, 2024"
    """
    if reference_date is None:
        reference_date = datetime.date.today()
    
    def get_ordinal(n):
        if 11 <= n % 100 <= 13:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"
    
    days_since_friday = (reference_date.weekday() - 4) % 7
    last_friday = reference_date - datetime.timedelta(days=days_since_friday)
    day_with_ordinal = get_ordinal(last_friday.day)
    formatted_date = f"Week ending {last_friday.strftime('%A')} {day_with_ordinal} {last_friday.strftime('%B')}, {last_friday.year}"
    
    return formatted_date

# Usage
formatted_week_label = get_week_ending_label()


# In[30]:


from datawrapper import Datawrapper

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Stock Sentiment Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=combined_df_unique)

# Get pressure column names
pressure_cols = [col for col in combined_df_unique.columns if col.startswith('sent_')]

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
                "title": "Sentiment",
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
for col in pressure_cols:
    properties["visualize"]["columns"][col] = {
        "type": "number",
        "width": 0.33,  # Updated to match working example
        "format": "0.000",
        "sparkline": {
            "color": "#18a1cd",
            "title": "sentiment_history",
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

current_date = datetime.datetime.now().strftime("%B %d, %Y")


# Add other visualization settings
properties["describe"] = {
    "intro": f"Analysis of news pressure for companies with trends over 60 days, as of {current_date}. Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/news-sentiment'>Sov.ai™ News</a> datasets.",
    "byline": "",
    "source-name": "Sentiment Data",
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
news_url = published_url[0]["url"]

# In[31]:


news_url = published_url[0]["url"]

# In[32]:


## Secrtorial Sentiment

df_sentiment_sect = sov.data("news/sentiment",full_history=True)

df_sentiment_sect = df_sentiment_sect.filter(["market_cap>50"])

df_sentiment_sect = df_sentiment_sect.reset_index()

# Calculate the start date (12 weeks before the max_date)
start_date = max_date - pd.Timedelta(weeks=12)

# Ensure 'date' is in datetime format
df_sentiment_sect['date'] = pd.to_datetime(df_sentiment_sect['date'])

df_sentiment_sect = df_sentiment_sect.merge(tickers_meta[["ticker","sector"]], on="ticker",how="left")

df_sentiment_sect = df_sentiment_sect.drop(columns=["ticker"]).groupby(["date","sector"]).mean().sort_index().reset_index()



df_wide = df_sentiment_sect.pivot(index='date', columns='sector', values='sentiment')


df_wide = df_wide.resample("D").ffill()  

df_wide = df_wide.rolling(20).mean()
 
df_wide = df_wide.tail(180).dropna()

df_ranks = df_wide.rank(pct=True, axis=0).reset_index()


df_ranks = df_ranks.set_index("date").rank(pct=True,axis=1)

df_ranks = df_ranks.rolling(20).mean().dropna()

df_ranks = df_ranks.rank(pct=True,axis=1)

df_ranks = df_ranks.reset_index()

# In[41]:


from datawrapper import Datawrapper

# First get the order of series from the last row of your dataframe
last_row = df_ranks.drop(columns=["date"]).iloc[-1]  # Get the last row
# Sort the columns by their values in descending order
final_positions = last_row.sort_values(ascending=False).index.tolist()

# Create color gradient
colors = [
    "#0066CC",  # Deep blue
    "#1975D1",
    "#4D94DB",
    "#66A3E0",
    "#80B2E6",
    "#CCE0F3",
    "#FFE5CC",
    "#FFCC99",
    "#FFB366",
    "#FF9933",
    "#FF8000",  # Deep orange
]

# Create color mapping based on final positions
color_mapping = {series: colors[i] for i, series in enumerate(final_positions)}

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")
# Create a new line chart
chart = dw.create_chart(
    title="Daily Sectorial Sentiment Rank",
    chart_type="d3-lines"
)

# Configure the visualization properties
metadata = {
    "visualize": {
        "dark-mode-invert": True,
        "interpolation": "natural",
        "x-grid": "ticks",
        "y-grid": "off",
        "opacity": 1,
        "scale-y": "linear",
        "base-color": 8,
        "line-width": 2,
        "label-colors": True,
        "label-margin": 138,
        "stack-to-100": False,
        "show-tooltips": True,
        "x-grid-format": "YYYY-MM-DD",
        "y-grid-format": "auto",
        "y-grid-labels": "auto",
        "plotHeightMode": "ratio",
        "plotHeightRatio": 0.62,
        "plotHeightFixed": 300,
        "color-by-column": True,
        "connector-lines": True,
        "y-grid-subdivide": True,
        "value-label-colors": True,
        "y-grid-label-align": "left",
        "tooltip": {
            "sticky": True,
            "enabled": True
        },
        "lines": {
            series: {
                "color": color_mapping[series],
                "symbols": {"size": 3, "style": "hollow", "enabled": True}
            } for series in final_positions
        }
    },
    "describe": {
        "byline": "Scraping various news related datasets",
        "intro": "The data has been collected since 2016 and allows us to identify daily changes in news sentiment accross sectors.",
        "source-name": "",
        "source-url": ""
    },
    "axes": {
        "x": "date"
    },
    "publish": {
        "embed-width": 628,
        "embed-height": 411,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        },
        "chart-height": 307
    }
}

# Update the chart with our configuration
dw.update_chart(chart['id'], metadata=metadata)
# Add the data to the chart
dw.add_data(chart['id'], data=df_ranks)
# Publish the chart
dw.publish_chart(chart['id'])
# Get the embed code

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)
sector_url = published_url[0]["url"]

# In[34]:


df_final_sentiment.head()

# In[35]:


df_sentiment = sov.data("news/sentiment_score", full_history=True)

df_smaller = df_sentiment[df_sentiment["calculation"]=="sentiment_score_median"].reset_index(drop=True).drop(columns=["calculation"]).set_index("date").rolling(60).mean().dropna().tail(90)

# In[36]:


import pandas as pd

# Revised mapping dictionary with 15 succinct themes
theme_mapping = {
    'Monetary Policy': [
        'monetary_policy',
        'monetary_policy_transmission',
        'interest_rates',
        'quantitative_easing',
        'central_bank_digital_currencies'
    ],
    'Fiscal Policy': [
        'fiscal_policy',
        'government_debt_deficit',
        'tax_policy',
        'spending_policy'
    ],
    'Growth Indicators': [
        'economic_growth',
        'business_cycles',
        'economic_forecasting_modeling'
    ],
    'Inflation': [
        'inflation',
        'price_stability',
        'cost_of_living'
    ],
    'Intl Trade & Finance': [
        'international_trade',
        'foreign_direct_investment',
        'foreign_exchange_markets',
        'trade_agreements',
        'international_monetary_system',
        'international_finance',
        'emerging_economies',
        'sanctions_embargoes',
        'international_development_aid'
    ],
    'Sectors': [
        'energy_resources',
        'healthcare_pharma',
        'real_estate_housing',
        'consumer_spending_retail',
        'manufacturing_industrial',
        'transportation_logistics',
        'agriculture_food',
        'aerospace_defense',
        'utilities_public_services',
        'mining_extraction',
        'chemicals_materials',
        'forestry_paper_products',
        'fishing_aquaculture',
        'textiles_apparel',
        'luxury_goods_services',
        'sports_entertainment',
        'media_publishing'
    ],
    'Technology': [
        'artificial_intelligence',
        'robotics_automation',
        'cybersecurity_data_privacy',
        'cryptocurrency_blockchain',
        'quantum_computing',
        'technology_innovation',
        'intellectual_property_patents',
        'nanotech_advanced_materials',
        'space_commercialization_exploration',
        'renewable_energy',
        'digital_economy',
        'financial_technology_fintech'
    ],
    'Financial Markets': [
        'financial_markets_investing',
        'insurance_risk_management',
        'private_equity_venture_capital',
        'sovereign_wealth_funds',
        'pension_funds',
        'hedge_funds',
        'exchange_traded_funds',
        'mergers_acquisitions',
        'initial_public_offerings',
        'bond_markets',
        'derivative_markets',
        'yield_curve',
        'credit_ratings',
        'high_frequency_trading',
        'algorithmic_trading',
        'robo_advisors'
    ],
    'ESG': [
        'environmental_sustainability',
        'climate_change',
        'renewable_energy',
        'circular_economy',
        'sustainable_development',
        'human_rights_business',
        'social_responsibility',
        'governance_standards'
    ],
    'Social Trends': [
        'demographic_shifts_aging',
        'poverty_alleviation',
        'education_human_capital',
        'income_inequality',
        'welfare_inequality',
        'labor_market',
        'labor_productivity',
        'education_services',
        'sharing_economy_gig_work'
    ],
    'Geopolitics': [
        'geopolitical_events',
        'sanctions_embargoes',
        'migration_remittances',
        'international_development_aid',
        'foreign_direct_investment',
        'foreign_exchange_markets'
    ],
    'Infrastructure': [
        'urbanization_city_planning',
        'infrastructure',
        'transportation_logistics',
        'utilities_public_services',
        'waste_management_recycling'
    ],
    'Labor Market': [
        'labor_market',
        'labor_productivity',
        'small_business_entrepreneurship',
        'consulting_business_services',
        'legal_services_regulations'
    ],
    'Regulations': [
        'consulting_business_services',
        'legal_services_regulations',
        'corporate_governance',
        'market_regulation',
        'antitrust_competition_policy'
    ],
    'Risk Management': [
        'insurance_risk_management',
        'financial_stability',
        'systemic_risk',
        'credit_ratings',
        'distressed_debt',
        'leveraged_buyouts',
        'short_selling'
    ]
}

# Example usage: Mapping DataFrame columns to themes
# Assuming you have a DataFrame `df` with relevant columns
# You can create a new column 'Theme' by mapping each existing column to its theme

# Sample DataFrame columns
df_columns = [
    'monetary_policy', 'interest_rates', 'tax_policy',
    'economic_growth', 'inflation', 'international_trade',
    'energy_resources', 'artificial_intelligence',
    'financial_markets_investing', 'environmental_sustainability',
    'demographic_shifts_aging', 'geopolitical_events',
    'infrastructure', 'labor_market', 'consulting_business_services',
    'insurance_risk_management'
]

# Create a reverse mapping from topic to theme
reverse_mapping = {}
for theme, topics in theme_mapping.items():
    for topic in topics:
        reverse_mapping[topic] = theme

# Function to map a column to its theme
def map_column_to_theme(column):
    return reverse_mapping.get(column, 'Uncategorized')

# Example DataFrame
df = pd.DataFrame(columns=df_columns)

# Apply the mapping to assign themes
df_mapped = df.columns.to_series().apply(map_column_to_theme).reset_index()
df_mapped.columns = ['Column', 'Theme']

print(df_mapped)


import pandas as pd

# Assuming you have your original dataframe 'df_smaller'
# Example:
# df_smaller = pd.read_csv('your_data.csv', parse_dates=['date'])
df_smaller = df_smaller.reset_index()
# Create a new dataframe to store the averaged themes
df_output = pd.DataFrame()
df_output['date'] = df_smaller['date']  # Preserve the date column

# Iterate through each theme and compute the mean
for theme, columns in theme_mapping.items():
    # Check if all columns exist in the dataframe
    missing_cols = [col for col in columns if col not in df_smaller.columns]
    if missing_cols:
        print(f"Warning: The following columns for theme '{theme}' are missing in df_smaller: {missing_cols}")
        # Optionally, you can choose to skip these columns or handle them differently
    # Compute the mean, skipping missing columns
    available_cols = [col for col in columns if col in df_smaller.columns]
    if available_cols:
        df_output[theme] = df_smaller[available_cols].mean(axis=1)
    else:
        df_output[theme] = pd.NA  # Assign NA if no columns are available

# Optionally, set 'date' as the index
df_output.set_index('date', inplace=True)

# Display the resulting dataframe
print(df_output.head())


df_ranks = df_output.rank(pct=True, axis=1).reset_index()


final_positions

# In[37]:


from datawrapper import Datawrapper

# First get the order of series from the last row of your dataframe
last_row = df_ranks.drop(columns=["date"]).iloc[-1]  # Get the last row
# Sort the columns by their values in descending order
final_positions = last_row.sort_values(ascending=False).index.tolist()

# Create color gradient
colors = [
    "#0066CC",  # Deep blue
    "#1975D1",
    "#3385D6",
    "#4D94DB",
    "#66A3E0",
    "#80B2E6",
    "#99C2E9",
    "#B2D1ED",
    "#CCE0F3",
    "#FFE5CC",
    "#FFCC99",
    "#FFB366",
    "#FF9933",
    "#FF9033",
    "#FF8000",  # Deep orange
]

# Create color mapping based on final positions
color_mapping = {series: colors[i] for i, series in enumerate(final_positions)}

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")
# Create a new line chart
chart = dw.create_chart(
    title="Daily Topic Sentiment Rank",
    chart_type="d3-lines"
)

# Configure the visualization properties
metadata = {
    "visualize": {
        "dark-mode-invert": True,
        "interpolation": "natural",
        "x-grid": "ticks",
        "y-grid": "off",
        "opacity": 1,
        "scale-y": "linear",
        "base-color": 8,
        "line-width": 2,
        "label-colors": True,
        "label-margin": 138,
        "stack-to-100": False,
        "show-tooltips": True,
        "x-grid-format": "YYYY-MM-DD",
        "y-grid-format": "auto",
        "y-grid-labels": "auto",
        "plotHeightMode": "ratio",
        "plotHeightRatio": 0.62,
        "plotHeightFixed": 300,
        "color-by-column": True,
        "connector-lines": True,
        "y-grid-subdivide": True,
        "value-label-colors": True,
        "y-grid-label-align": "left",
        "tooltip": {
            "sticky": True,
            "enabled": True
        },
        "lines": {
            series: {
                "color": color_mapping[series],
                "symbols": {"size": 3, "style": "hollow", "enabled": True}
            } for series in final_positions
        }
    },
    "describe": {
        "byline": "50k websites scraped daily since 2016",
        "intro": "Data derived from daily analysis of 50,000 websites using natural language processing and machine learning algorithms to identify and track topic sentiment.",
        "source-name": "",
        "source-url": ""
    },
    "axes": {
        "x": "date"
    },
    "publish": {
        "embed-width": 628,
        "embed-height": 411,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        },
        "chart-height": 307
    }
}

# Update the chart with our configuration
dw.update_chart(chart['id'], metadata=metadata)
# Add the data to the chart
dw.add_data(chart['id'], data=df_ranks)
# Publish the chart
dw.publish_chart(chart['id'])
# Get the embed code

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)
thematic_url = published_url[0]["url"]

# In[40]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": f"Ticker, Sector, and Topic Sentiment",
        "content": (
            "Tracking ticker-level sentiment allows us to aggregate sentiment to the sectorial level, and we also"
            " explicity track 98 different themes that we aggregate to main themes for simplicity."
            
        ),
        "url": news_url,
        "list": None
    },
    "section_2": {
        "heading": None,
        "content": (
            "Here we are tracking the sentiment at a sectorial level which is helpful to sectorial sentiment shifts"
            
        ),
        "url": sector_url,
        "list": None
    },
    "section_3": {
        "heading": None,
        "content": (
            "We also track numerous themes that we aggreagate in a handful of main themes, the detailed themes are available in the SDK."
            
        ),
        "url": thematic_url,
        "list": None
    },
    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

