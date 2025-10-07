#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[2]:


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


# In[3]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")

df_contracts = sov.data("spending/contracts",  purge_cache=True,verbose=True, full_history=True)

import pandas as pd

# Ensure 'date' column is datetime
df_contracts['date'] = pd.to_datetime(df_contracts['date'])

# Filter out entries within one week before the max date
df_contracts_filter = df_contracts[df_contracts['date'] > (df_contracts['date'].max() - pd.Timedelta(days=7))]


import pandas as pd
import numpy as np

# ---------------------------------------
# 5. Define and Perform Aggregations by Ticker
# ---------------------------------------

# Define the aggregation functions for each metric, including 'recipient_name' using 'first' and 'date' using 'min'
aggregation_functions = {
    'potential_total_value_of_award': 'sum',
    'total_federal_action_obligation': 'sum',
    'obligation_value_difference': 'mean',
    'contract_award_unique_key': 'count',
    'performance_duration': 'mean',
    'awards_past_year': 'sum',
    'transactions_per_award': 'sum',
    'time_to_start_performance': 'mean',
    'extension_days_available': 'sum',
    'modification_number': 'sum',
    'recipient_name': 'first',  # Include recipient_name using 'first'
    'date': 'min'  # Include the minimum date by ticker
}

# Group by 'ticker' and perform aggregations
summary_df = df_contracts_filter.groupby('ticker').agg(aggregation_functions).reset_index()

# Rename columns for clarity
summary_df.rename(columns={
    'ticker': 'Ticker',
    'potential_total_value_of_award': 'Total_Potential_Value',
    'total_federal_action_obligation': 'Total_Federal_Obligation',
    'obligation_value_difference': 'Avg_Obligation_Value_Diff',
    'contract_award_unique_key': 'Total_Awards',
    'performance_duration': 'Avg_Performance_Duration',
    'awards_past_year': 'Total_Awards_Past_Year',
    'transactions_per_award': 'Total_Transactions',
    'time_to_start_performance': 'Avg_Time_to_Start_Performance',
    'extension_days_available': 'Total_Extension_Days',
    'modification_number': 'Total_Modifications',
    'recipient_name': 'Recipient_Name',
    'date': 'Min_Date'  # Rename the minimum date column
}, inplace=True)

# Handle potential infinities or NaNs resulting from aggregation
summary_df.replace([np.inf, -np.inf], np.nan, inplace=True)
summary_df.fillna(0, inplace=True)

# Sort the summary by Total Potential Value in descending order
summary_df.sort_values(by='Total_Potential_Value', ascending=False, inplace=True)

# ---------------------------------------
# 6. Calculate Obligation Percentage
# ---------------------------------------

# Correct the Obligation_Percentage calculation
# Assuming you want to calculate Total_Federal_Obligation as a percentage of Total_Potential_Value
summary_df["Obligation_Percentage"] = (summary_df["Total_Federal_Obligation"] / summary_df["Total_Potential_Value"]) * 100

# Handle division by zero if Total_Potential_Value is zero
summary_df["Obligation_Percentage"] = summary_df["Obligation_Percentage"].replace([np.inf, -np.inf], np.nan)
summary_df["Obligation_Percentage"] = summary_df["Obligation_Percentage"].fillna(0)

# ---------------------------------------
# 7. Drop Unnecessary Columns
# ---------------------------------------

# Drop the specified columns
summary_df = summary_df.drop(columns=[
    "Total_Transactions",
    "Avg_Time_to_Start_Performance",
    "Total_Extension_Days",
    "Total_Modifications"
])


summary_df = summary_df.rename(columns={
    'Total_Potential_Value': 'Size Mn',
    'Total_Federal_Obligation': 'Obligation',
    'Total_Awards': 'Awards',
    'Avg_Performance_Duration': 'Duration',
    'Total_Awards_Past_Year': 'Previous',
    'Recipient_Name': 'Recipient',
    'Min_Date': 'Date',
    'Obligation_Percentage': 'Obliged %',
    'Avg_Obligation_Value_Diff': 'Value_Diff'
})
summary_df = summary_df[["Ticker","Date","Size Mn","Obligation","Obliged %","Awards","Previous","Duration"]]
# Remove rows with negative obligations and zero Size
summary_df = summary_df[
    (summary_df['Obligation'] > 0) &
    (summary_df['Size Mn'] > 0)
]

# Convert Size and Obligation to millions
summary_df['Size Mn'] = (summary_df['Size Mn'] / 1_000_000).round(3)
summary_df['Obligation'] = (summary_df['Obligation'] / 1_000_000).round(3)

# Round percentage to 2 decimal places
summary_df['Obliged %'] = summary_df['Obliged %'].round(2)

# Round Duration and Past_Awards to integer
summary_df['Duration'] = summary_df['Duration'].round(0).astype(int)
summary_df['Previous'] = summary_df['Previous'].round(0).astype(int)
# Format the data
# Add hyperlinks to tickers and drop date column
summary_df['Ticker'] = summary_df['Ticker'].apply(
    lambda x: f"[{x}](https://finance.yahoo.com/quote/{x})"
)

# In[4]:


summary_df

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


# In[6]:


from datawrapper import Datawrapper

# Initialize Datawrapper with your API token
dw = Datawrapper(access_token="your_token")


df = summary_df.copy()

# Create the chart
chart = dw.create_chart(
    title="Government Contract Awards",
    chart_type="tables"
)

# Add the data
dw.add_data(chart['id'], data=df)

# Configure visualization properties
properties = {
    "visualize": {
        "dark-mode-invert": True,
        "perPage": 20,
        "columns": {
            "Ticker": {
                "align": "left",
                "title": "Stock",
                "width": "100",
                "markdown": True
            },
            "Date": {
                "title": "Date",
                "width": "100",
                "format": "YYYY-MM-DD"
            },
            "Size Mn": {
                "title": "Contract Size ($M)",
                "width": "120",
                "format": "$0,0.000"
            },
            "Obligation": {
                "title": "Obligated ($M)",
                "width": "120",
                "format": "$0,0.000"
            },
            "Obliged %": {
                "title": "% Obligated",
                "width": 0.27,
                "format": "0.0%",
                "showAsBar": True,
                "fixedWidth": True
            },
            "Awards": {
                "title": "Awards Today",
                "width": "100",
                "format": "0"
            },
            "Previous": {
                "title": "Previous Awards",
                "width": "120",
                "format": "0"
            },
            "Duration": {
                "title": "Duration (Days)",
                "width": "120",
                "format": "0",
                "showAsBar": True
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
        "firstColumnIsSticky": True
    },
    "describe": {
        "intro": ("Government contract awards and obligations by company, showing contract sizes, obligation rates, and historical context."
                 f" {formatted_week_label}."
                 " Derived from <a href='https://docs.sov.ai/realtime-datasets/equity-datasets/government-contracts'>Sov.ai™ Government Contracts</a> datasets."),
        "byline": "",
        "source-name": "Government Contract Data",
        "hide-title": False
    },
    "publish": {
        "embed-width": 1000,
        "embed-height": 800,
        "blocks": {
            "logo": {"enabled": False},
            "embed": False,
            "download-pdf": False,
            "download-svg": False,
            "get-the-data": True,
            "download-image": False
        },
        "chart-height": 700
    }
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

# In[9]:


from datetime import datetime
# Define title
page_title = "Predict a Mockingbird"

# Define content sections using the content_sections dictionary
content_sections = {
    "section_1": {
        "heading": "Government Spending",
        "content": (
            "This data tracks recent announcements of contracts being made by the government, in many quarters the government"
            " is responsible for more than 40% of the national expenditures and are important source of revenue for many companies."
            
        ),
        "url": published_url[0]["url"],
        "list": None
    }

    # Add more sections as needed
}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

