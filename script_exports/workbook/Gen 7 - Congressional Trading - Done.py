#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[8]:


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


# In[1]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")


# In[2]:


tickers_meta = pd.read_parquet("data/tickers.parq")

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


import pandas as pd
from datetime import timedelta
import requests
import numpy as np

df_congress = sov.data("congress")
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


# In[5]:


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


# In[7]:


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
    "intro": ("Congressional stock trades with 90-day historical price performance."
                 f" {formatted_week_label}."
                 " Derived from <a href='https://docs.sov.ai/'>Sov.ai™ Congress</a> datasets."),
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

# In[12]:


# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Congressional Trading Tracker",
        "content": (
            "This tracks filings and records in the house and senate (both houses of congress) with the underlying theory that they"
            " are perhaps more informed than the average investor"
            
        ),
        "url": published_url[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

