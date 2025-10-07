#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[1]:


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
    current_date = datetime.now().strftime("%Y-%m-%d")
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


# In[2]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

# In[3]:


tickers_meta = pd.read_parquet("data/tickers.parq")

# In[4]:


df_wiki = sov.data("wikipedia/views", full_history=True)

# In[5]:



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

# In[6]:


df_org = sov.data("wikipedia/views", full_history=True)

# In[7]:


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


# In[8]:


df_wiki_expanded[["search","last","week change","long","day change","short"]] = df_wiki_expanded[["search","last","week change","long","day change","short"]]*100

# In[9]:


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


combined_df_unique = combined_df_unique.sample(100)

combined_df_unique = combined_df_unique.drop(columns=["long"])

combined_df_unique = combined_df_unique.rename(columns={"short":"long"})

combined_df_unique

# In[14]:


from datawrapper import Datawrapper

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
from datetime import datetime

current_date = datetime.now().strftime("%B %d, %Y")

# Add other visualization settings
properties["describe"] = {
    "intro": f"""Analysis of Wikipedia page views for stocks with historical pressure trends over 60 days, as of {current_date}. 
    This analysis tracks market sentiment through public interest patterns. 
    Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/wikipedia-views'>Sov.ai™ Wiki</a> datasets. """,
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

# In[15]:


wiki_url = published_url[0]["url"]

# In[16]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": f"Wikipedia Search Pressure",
        "content": (
            "This model uses an algorithm to discover early search pressure on wikipedia pages."
            " This signal is stronger than most retail signals because it reflexts the start of a true research process."
            
        ),
        "url": wiki_url,
        "list": None
    },

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

