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

tickers_meta = pd.read_parquet("data/tickers.parq")

df_earn_surp = sov.data("earnings/surprise")

import requests
import pandas as pd
from datetime import datetime, timedelta

# Your API Key
API_KEY = 'your_key'

def get_earnings_calendar(api_key):
    """
    Fetches the earnings calendar for all US stocks over the next two weeks.

    Parameters:
        api_key (str): Your Financial Modeling Prep API key.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the earnings calendar.
    """
    # Get today's date
    today = datetime.today()
    # Calculate the date two weeks from today
    two_weeks_later = today + timedelta(weeks=2)

    # Format dates as YYYY-MM-DD
    from_date = today.strftime('%Y-%m-%d')
    to_date = two_weeks_later.strftime('%Y-%m-%d')

    # Construct the API URL
    url = (
        f'https://financialmodelingprep.com/api/v3/earning_calendar'
        f'?from={from_date}&to={to_date}&apikey={api_key}'
    )

    try:
        # Make the GET request to the API
        response = requests.get(url)
        # Raise an exception if the request was unsuccessful
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')  # HTTP error
        return pd.DataFrame()  # Return empty DataFrame on error
    except Exception as err:
        print(f'An error occurred: {err}')  # Other errors
        return pd.DataFrame()  # Return empty DataFrame on error

    # Parse the JSON response
    earnings_data = response.json()

    # Check if the response contains data
    if not earnings_data:
        print('No earnings data found for the specified date range.')
        return pd.DataFrame()

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(earnings_data)

    return df

earnings_calendar_df = get_earnings_calendar(API_KEY)


earnings_calendar_df = earnings_calendar_df.merge(df_earn_surp.reset_index().drop(columns=["date"]), left_on=["symbol"], right_on="ticker", how="left")

earnings_calendar_df

earnings_calendar_df = earnings_calendar_df.dropna(subset=["surprise_probability"])[["ticker","date","time", "epsEstimated","surprise_probability", "fiscalDateEnding","updatedFromDate"]].sort_values("surprise_probability")

# Uppercase the time column
earnings_calendar_df['time'] = earnings_calendar_df['time'].str.upper()

# Rename the last 4 columns
earnings_calendar_df = earnings_calendar_df.rename(columns={
   'epsEstimated': 'eps_est',
   'surprise_probability': 'surprise_prob', 
   'fiscalDateEnding': 'quarter_end',
   'updatedFromDate': 'last_update'
})

earnings_calendar_df = earnings_calendar_df.sort_values("date")

# First, let's clean and prepare the data
earnings_df = earnings_calendar_df.copy()

# Format the date column
earnings_df['date'] = pd.to_datetime(earnings_df['date']).dt.strftime('%Y-%m-%d')

# Create a Yahoo Finance URL for each ticker
earnings_df['ticker'] = earnings_df['ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

# Round numeric columns
if 'eps_est' in earnings_df.columns:
    earnings_df['eps_est'] = earnings_df['eps_est'].round(3)
if 'surprise_prob' in earnings_df.columns:
    earnings_df['surprise_prob'] = (earnings_df['surprise_prob'] * 100).round(1)


# In[4]:


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


# In[5]:


from datawrapper import Datawrapper

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create chart
chart = dw.create_chart(
    title="Upcoming Earnings Announcements",
    chart_type="tables"
)

# Add data
dw.add_data(chart['id'], data=earnings_df)

# Configure visualization
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "columns": {
            "ticker": {
                "title": "Stock",
                "width": "100",
                "align": "left",
                "markdown": True,
                "fixedWidth": False
            },
            "date": {
                "title": "Date",
                "width": "120",
                "fixedWidth": False
            },
            "time": {
                "title": "Time",
                "width": "80",
                "fixedWidth": False
            },
            "eps_est": {
                "title": "EPS Est.",
                "format": "$0.00",
                "width": "100",
                "fixedWidth": False,
                "showAsBar": False,
                "heatmap": {"enabled": False}
            },
            "surprise_prob": {
                "title": "Surprise Prob.",
                "format": "0.0%",
                "width": "0.27",
                "showAsBar": True,
                "borderLeft": "1px",
                "fixedWidth": True,
                "barColorNegative": "#ff4444",
                "barColorPositive": "#44bb77",
                "style": {"fontSize": 1.13}
            },
                "quarter_end": {
                "borderLeft": "1px",
            },
        },
        "header": {
            "style": {
                "bold": True,
                "fontSize": 0.9,
                "color": "#494949",
                "italic": False,
                "background": False
            },
            "borderTop": "none",
            "borderBottom": "2px",
            "borderTopColor": "#333333",
            "borderBottomColor": "#333333"
        },
        "pagination": {
            "enabled": True,
            "position": "bottom",
            "pagesPerScreen": 10
        },
        "perPage": 15,
        "striped": True,
        "markdown": True,
        "showHeader": True,
        "compactMode": True,
        "firstRowIsHeader": False,
        "firstColumnIsSticky": True,
        "searchable": True,
        "search": {
            "enabled": True,
            "placeholder": "Search stocks..."
        }
    },
    "describe": {
         "intro": (f"Expected earnings announcements for the next two weeks with estimated EPS and surprise probability. Use the search box to find specific stocks.. {formatted_week_label}"
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets'>Sov.ai™ Surprise</a> datasets."),
               
        
        "byline": "",
        "source-name": "Earnings Data",
        "hide-title": False
    },
    "publish": {
        "embed-width": 700,
        "embed-height": 1159,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": False,
            "download-image": True
        },
        "autoDarkMode": False,
        "chart-height": 1044,
        "force-attribution": False
    }
}

# Update and publish chart
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[7]:


from datetime import datetime
# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Earnings Surprise Prediction",
        "content": (
            "Here we are interested in what companies are expected to experience an earnings surprise in the near future."
            " The benefit of predicting earnings surprises is that you can act before the market reacts."
            
        ),
        "url": published_url[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

