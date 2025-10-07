#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[25]:


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


# In[11]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")

df_short = sov.data("short/over_shorted")

df_short = sov.data("short/over_shorted")

df_short = df_short.drop(columns=["number_of_shares","total_revenue","volume","days_to_cover"])

df_shorted = pd.concat([df_short.nsmallest(30, 'over_shorted'), df_short.nlargest(30, 'over_shorted')])

df_shorted = df_shorted.sort_values("over_shorted",ascending=False).reset_index()

df_shorted[["over_shorted","over_shorted_chg","short_percentage","short_prediction"]] = df_shorted[["over_shorted","over_shorted_chg","short_percentage","short_prediction"]] *100

df_shorted["short_interest"] = df_shorted["short_interest"] /1000

df_shorted = df_shorted.rename(columns={"over_shorted_chg":"change","short_percentage":"actual","short_prediction":"expected","short_interest":"short_int"})

max_date = df_shorted["date"].max()
formatted_date = max_date.strftime('%B %d, %Y')


df_shorted = df_shorted.drop(columns=["date"])

# In[12]:


df_shorted['ticker'] = df_shorted['ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

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


# In[28]:


from datawrapper import Datawrapper

# Initialize Datawrapper with the access token
dw = Datawrapper(access_token="your_token")


# Create a new chart
chart = dw.create_chart(
    title="Stock Short Interest Analysis",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=df_shorted)

# Configure the visualization properties
metadata = {
    "visualize": {
        "columns": {
            "ticker": {
                "title": "Ticker", 
                "align": "left",
                "width": "100",
                "markdown": True,  # Enable markdown for clickable links
                "bold": True
            },
            "over_shorted": {
                "title": "Over Shorted", 
                "format": "0.0%",
                "showAsBar": True,
                "barColorNegative": 7,
                "width": "120"
            },
            "change": {
                "title": "Change", 
                "format": "+0.0%",
                "width": "100"
            },
            "actual": {
                "title": "Actual Short %", 
                "format": "0.0%",
                "width": "120"
            },
            "expected": {
                "title": "Expected Short %", 
                "format": "0.0%",
                "width": "120"
            },
            "market_cap": {
                "title": "Market Cap (M)", 
                "format": "$ 0,0.0",
                "width": "120"
            },
        },
        "sortBy": "over_shorted",
        "markdown": True,  # Enable markdown globally
        "sortDirection": "desc",
        "perPage": 15,  # Added this setting
        "pagination": {
            "enabled": True,
            "position": "bottom",
            "rowsPerPage": 15  # Updated to 15
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 1.1
            },
            "borderBottom": "2px",
            "borderBottomColor": "#333333"
        },
        "showHeader": True
    },
    "describe": {
        "intro": (f"Analysis of most shorted stocks as of {formatted_date}. The table shows over-shorted positions, their recent changes, and actual vs expected short percentages."
            " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/short-selling'>Sov.ai™ Short</a> datasets.")
        ,
        "byline": "",
        "source-name": "Market Data",
        "source-url": ""
    },
    "publish": {
        "embed-width": 648,
        "embed-height": 776
    }
}

# Update the chart with the metadata
dw.update_chart(
    chart_id=chart['id'],
    metadata=metadata
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the URL of the published chart
chart_urls = dw.get_chart_display_urls(chart['id'])
print("\nChart URLs:", chart_urls)

# In[16]:


short_url = chart_urls[0]["url"]

# In[29]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

df_short_vol_hist = sov.data("short/volume", full_history=True)

df_short_vol_hist["short_ratio"] = df_short_vol_hist["short_volume"]/df_short_vol_hist["total_volume"]

df_short_vol_hist.query("ticker == 'AAPL'")

df_short_vol_hist.tail()

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



df_short_vol_levels[["sr_curr","sr_prev","sr_diff","sr_ex","sr_ret","sr_int","sr_mm"]] = df_short_vol_levels[["sr_curr","sr_prev","sr_diff","sr_ex","sr_ret","sr_int","sr_mm"]]*100
df_short_vol_levels




tickers_meta = pd.read_parquet("data/tickers.parq")

df_short_vol_levels = df_short_vol_levels.set_index(["ticker","date"])

df_short_vol_levels = df_short_vol_levels.filter(["market_cap>100"])

# Sort by 'sr_diff' and get top 30 and bottom 30, concatenate them
df_sorted = df_short_vol_levels.sort_values("sr_diff").reset_index()
df_top_bottom = pd.concat([df_sorted.head(150), df_sorted.tail(150)], ignore_index=True)

max_date = df_top_bottom["date"].max()
formatted_date = max_date.strftime('%B %d, %Y')
df_top_bottom = df_top_bottom.drop(columns=["date"])

df_top_bottom = df_top_bottom.drop(columns=["sr_ret","sr_int","sr_mm"])

df_top_bottom = df_top_bottom.rename(columns={"sr_curr":"short_ratio"})

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
        "intro": ("Analysis of largest short ratio changes. The table shows short volume, total volume, and short ratio (SR) metrics including current SR, previous SR, and the change between them."
                 f" {formatted_week_label}."
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/short-selling'>Sov.ai™ Short</a> datasets."),
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

# In[23]:


volume_url = chart_urls[0]["url"]

# In[24]:


volume_url

# In[26]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": f"Short Interest Analysis - {datetime.now().strftime('%Y-%m-%d')}",
        "content": (
            "Our model identifies both overshorted companies (large values) prime for potential squeezes and undershorted stocks (low values) likely to face increased shorting pressure. The model updates every two weeks as short interest files are made available."
        ),
        "url": short_url,
        "list": None
    },
    "section_2": {
        "heading": None,
        "content": (
            "By the time the short-interest file becomes available, the market may have already reacted. "
            "( Monitoring changes in short volume, however, could provide an earlier indication of shifts in market sentiment. "
        ),
        "url": volume_url,
        "list": None
    },

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

