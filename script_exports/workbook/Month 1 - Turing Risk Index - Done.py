#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[ ]:


# !pip install pandas_datareader

# In[46]:


df_risks.head()

# In[11]:


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


# In[12]:


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


# In[13]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token") 

df_risks = sov.data("risks")

df_risks_agg = sov.compute('risk-aggregates', df=df_risks); df_risks_agg.tail()


# In[14]:


from datawrapper import Datawrapper
import pandas as pd


# Prepare the data first to get the date range
df = df_risks_agg.tail(3600)


# import pandas_datareader as pdr

# # Get recession data
# recession = pdr.get_data_fred('USRECD', start="1960-01-01")
# recession = recession.astype(float)
# recession = recession + 1

# min_date = df.index.min()
# max_date = df.index.max()

# print(f"Data range: {min_date} to {max_date}")

# # Filter recession data to match the data range
# recession = recession[recession.index >= min_date]
# recession = recession[recession.diff().abs() == 1].dropna()

# # Get recession periods
# start = recession[recession == 2].dropna().index
# end = recession[recession == 1].dropna().index

# # Convert to lists and handle any missing end dates
# start = start.to_list()
# end = end.to_list()
# if len(start) > len(end):
#     end.append(max_date)

# # Print recession periods for verification
# print("\nRecession periods:")
# for s, e in zip(start, end):
#     print(f"{s} to {e}")

# # Create range annotations for recessions
# range_annotations = []
# for s, e in zip(start, end):
#     range_annotations.append({
#         "x0": s.strftime('%Y-%m-%d'),
#         "x1": e.strftime('%Y-%m-%d'),
#         "y0": 0,
#         "y1": 100,
#         "color": "rgba(128, 128, 128, 0.15)",  # Lighter gray
#         "type": "x",
#         "display": "range"
#     })


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Key Risk Metrics",
    chart_type="d3-lines"
)

# Prepare chart data
chart_data = df[['TURING_RISK', 'MARKET_RISK', 'BUSINESS_RISK', 'POLITICAL_RISK']].copy()
chart_data['Risk_Threshold'] = 30
chart_data.index = pd.to_datetime(chart_data.index).strftime('%Y-%m-%d')
chart_data = chart_data.reset_index()
chart_data = chart_data.rename(columns={'index': 'Date'})

# Add the data to the chart
dw.add_data(chart['id'], data=chart_data)


# Configure the visualization
metadata = {
    "visualize": {
        "dark-mode-invert": True,
        "interpolation": "linear",
        "lines": {
            "TURING_RISK": {
                "color": "#ff0000",
                "width": "style3",
                "visible": True,
                "valueLabels": {
                    "enabled": True,
                    "maxInnerLabels": 8  # Will automatically place labels at key points
                }
            },
            "MARKET_RISK": {
                "color": "#15607a",
                "width": "style3"
            },
            "BUSINESS_RISK": {
                "color": "#15607a",
                "width": "style3"
            },
            "POLITICAL_RISK": {
                "color": "#15607a",
                "width": "style3"
            },
            "Risk_Threshold": {
                "dash": "style1",
                "color": "#FFA500",  # Light pink color
                "width": "style1",
                "type": "line",
                "stroke-dasharray": "2,2"
            }
        },
        "y-grid": "on",
        "x-grid": "on",
        "legend": {
            "enabled": True,
            "position": "top"
        },
        "tooltip": {
            "enabled": True,
            "show-title": True
        },
        "x-axis": {
            "tick-format": "%Y-%m-%d"
        },
        "y-grid-format": "0,0.[00]",
        "value-labels-format": "0.0",
        "label-colors": True,
        "show-tooltips": True,
    },
    "describe": {
"intro": """<p>The TURING RISK indicator (<span style='color: #ff0000;'>red line</span>) provides key signals. Above 30 (<span style='color: #FFA500;'>orange line</span>): Indicates heightened risk</p>
        <p>When trending downward: Signals improving conditions. </p>""",
        "byline": "",
        "source-name": "Federal Reserve Economic Data (FRED)",
        "number-format": "0.0",
    },
    "publish": {
        "chart-height": 400,
        "embed-width": 700,
    }
}

# Update and publish
try:
    result = dw.update_chart(chart['id'], metadata=metadata)
    # print("\nUpdate result: Chart updated successfully")
except Exception as e:
    print("\nError updating chart:", str(e))

# Publish the chart
try:
    publish_result = dw.publish_chart(chart['id'])
    # print("Publish result: Chart published successfully")
    
    # Print the response for debugging
    # print("\nPublish response:", publish_result)
except Exception as e:
    print("Error publishing chart:", str(e))

# Get the URL
try:
    published_url_lines = dw.get_chart_display_urls(chart['id'])
    print("\nChart URL:", published_url)
except Exception as e:
    print("Error getting URL:", str(e))

# In[15]:


df_business = sov.data("risks/business"); df_business.tail()
# Simply remove underscores and keep original names
filtered_df = df_business.drop(['NEWS_SENT_NEG', 'NAIIM_NEG'], axis=1).tail(180)
filtered_df.columns = [col.replace('_', ' ') for col in filtered_df.columns]

# In[33]:


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Business Risk Indicators",
    chart_type="multiple-lines",
    data=filtered_df.reset_index()
)

metadata = {
    "visualize": {
        "gridLayout": "fixedCount",
        "gridColumnCount": 7,
        "gridRowHeightMode": "fixed",
        "gridRowHeightFixed": 120,
        "gridColumnMinWidth": "auto",
        "interpolation": "monotone-x",
        "y-grid": "on",
        "x-grid": "off",
        "independentYScales": True,
        "lines": {
            col: {
                "color": "#34495E",
                "width": "style1",
                "labelSize": 0.5
            }
            for col in filtered_df.columns
        },
        "tooltip-x-format": "MMM DD",
        "tooltip-number-format": "0,0.[0]",
        "legend": {
            "enabled": False
        },
        "show-tooltips": True,
        "syncMultipleTooltips": True,
        "y-grid-labels": "right",
        "y-grid-format": "0,0.[0]",
        "label-margin": 0,
        "fontSize": 0.5
    },
    "describe": {
        "intro": "Multi-indicator dashboard tracking business and economic risk metrics.",
        "source-name": "Sovereign Model",
        "byline": f"Updated: {filtered_df.index[-1].strftime('%B %d, %Y')}",
    },
    "publish": {
        "embed-width": 1200,
        "chart-height": 800,
        "blocks": {
            "logo": {"enabled": False},
            "get-the-data": False
        }
    }
}

# Update and publish
dw.update_chart(chart['id'], metadata=metadata)
dw.publish_chart(chart['id'])

published_url_business= dw.get_chart_display_urls(chart['id']); published_url_business

# In[18]:


df_market = sov.data("risks/market"); df_market.tail()
# Simply remove underscores and keep original names
filtered_df = df_market.tail(180)
filtered_df.columns = [col.replace('_', ' ') for col in filtered_df.columns]

# In[36]:


# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Market Risk Indicators",  # Updated title
    chart_type="multiple-lines",
    data=filtered_df.reset_index()
)

metadata = {
    "visualize": {
        "gridLayout": "fixedCount",
        "gridColumnCount": 7,
        "gridRowHeightMode": "fixed",
        "gridRowHeightFixed": 120,
        "gridColumnMinWidth": "auto",
        "interpolation": "monotone-x",
        "y-grid": "on",
        "x-grid": "off",
        "independentYScales": True,
        "lines": {
            col: {
                "color": "#34495E",
                "width": "style1",
                "labelSize": 0.5
            }
            for col in filtered_df.columns
        },
        "tooltip-x-format": "MMM DD",
        "tooltip-number-format": "0,0.[0]",
        "legend": {
            "enabled": False
        },
        "show-tooltips": True,
        "syncMultipleTooltips": True,
        "y-grid-labels": "right",
        "y-grid-format": "0,0.[0]",
        "label-margin": 0,
        "fontSize": 0.5
    },
    "describe": {
        "intro": "Multi-indicator dashboard tracking market risk metrics.",  # Updated description
        "source-name": "Sovereign Model",
        "byline": f"Updated: {filtered_df.index[-1].strftime('%B %d, %Y')}",
    },
    "publish": {
        "embed-width": 1200,
        "chart-height": 800,
        "blocks": {
            "logo": {"enabled": False},
            "get-the-data": False
        }
    }
}

# Update and publish
dw.update_chart(chart['id'], metadata=metadata)
dw.publish_chart(chart['id'])

published_url_market= dw.get_chart_display_urls(chart['id']); published_url_market

# In[45]:


import pandas as pd
import numpy as np
df_risks = sov.data("risks")

df_two = df_risks[["TURING_RISK","SP500"]].tail(90)
# Calculate indexed values and MA
indexed_data = df_two.copy()
indexed_data['TURING_RISK'] = indexed_data['TURING_RISK'] / indexed_data['TURING_RISK'].iloc[0] * 100
indexed_data['SP500'] = indexed_data['SP500'] / indexed_data['SP500'].iloc[0] * 100
indexed_data['TURING_MA3'] = indexed_data['TURING_RISK'].rolling(window=7).mean()

# Determine current regime
last_risk = indexed_data['TURING_RISK'].iloc[-1]
last_ma = indexed_data['TURING_MA3'].iloc[-1]
current_signal = "SELL/HOLD" if last_risk > last_ma else "BUY/HOLD"

# Find periods for both high and low risk
high_risk_periods = []
low_risk_periods = []
high_start_date = None
low_start_date = None

for i in range(len(indexed_data)):
    if i > 0:  # Skip first row since we're comparing with previous
        # High risk period logic
        if indexed_data['TURING_RISK'].iloc[i] > indexed_data['TURING_MA3'].iloc[i]:
            if high_start_date is None:
                high_start_date = indexed_data.index[i]
            if low_start_date is not None:
                low_risk_periods.append({
                    'start': low_start_date,
                    'end': indexed_data.index[i]
                })
                low_start_date = None
        # Low risk period logic
        else:
            if low_start_date is None:
                low_start_date = indexed_data.index[i]
            if high_start_date is not None:
                high_risk_periods.append({
                    'start': high_start_date,
                    'end': indexed_data.index[i]
                })
                high_start_date = None

# Handle end of data periods
if high_start_date is not None:
    high_risk_periods.append({
        'start': high_start_date,
        'end': indexed_data.index[-1]
    })
if low_start_date is not None:
    low_risk_periods.append({
        'start': low_start_date,
        'end': indexed_data.index[-1]
    })

# Reset index for Datawrapper
indexed_data = indexed_data.reset_index()

# Format last date nicely
last_date = indexed_data['Date'].iloc[-1].strftime('%B %d, %Y')

# Create chart
dw = Datawrapper(access_token="your_token")

chart = dw.create_chart(
    title=f"Turing Risk Regimes - Short Crossover",
    chart_type="d3-lines",
    data=indexed_data
)

# Define a harmonious color palette at the top
COLORS = {
    'risk_line': "#34495E",      # Elegant dark blue for Turing Risk
    'sp500': "#2ECC71",          # Clear green for S&P 500
    'ma_line': "#95A5A6",        # Subtle gray for MA
    'high_risk': "#E74C3C",      # Clear red for high risk
    'low_risk': "#27AE60",       # Deep green for low risk
}

# [Previous code remains the same until range_annotations]

# Create range annotations with updated colors
range_annotations = [
    {
        "x0": str(period['start']),
        "x1": str(period['end']),
        "y0": 0,
        "y1": max(indexed_data['TURING_RISK'].max(), indexed_data['SP500'].max()) * 1.1,
        "type": "x",
        "color": COLORS['high_risk'],
        "display": "range",
        "opacity": 13,
        "strokeType": "solid",
        "strokeWidth": 1
    }
    for period in high_risk_periods
] + [
    {
        "x0": str(period['start']),
        "x1": str(period['end']),
        "y0": 0,
        "y1": max(indexed_data['TURING_RISK'].max(), indexed_data['SP500'].max()) * 1.1,
        "type": "x",
        "color": COLORS['low_risk'],
        "display": "range",
        "opacity": 8,
        "strokeType": "solid",
        "strokeWidth": 1
    }
    for period in low_risk_periods
]

metadata = {
    "visualize": {
        "y-grid": "on",
        "interpolation": "linear",
        "line-width": 2,
        "lines": {
            "TURING_RISK": {
                "color": COLORS['risk_line'],
                "width": "style3",
                "label": "Turing Risk Index"
            },
            "SP500": {
                "color": COLORS['sp500'],
                "width": "style3",
                "label": "S&P 500 (Indexed)"
            },
            "TURING_MA3": {
                "color": COLORS['ma_line'],
                "width": "style3",
                "stroke-dash": "4",
                "label": "7-day Moving Average"
            }
        },
        "range-annotations": range_annotations,
        "tooltip-x-format": "YYYY-MM-DD",
        "tooltip-number-format": "0,0.[00]",
        "legend": {
            "enabled": True,
            "position": "right"
        }
    },
    "describe": {
        "intro": f"Current Signal: <span style='color: {(COLORS['high_risk'] if current_signal == 'SELL/HOLD' else COLORS['low_risk'])}'><strong>{current_signal}</strong></span><br/><br/>Signals are generated when the Turing-Risk-Index crosses its 7-day moving average, indicating changes in the short-term regime.",
        "source-name": "Sovereign Model",
        "byline": f"Last updated: {last_date}",
        "aria-description": f"Market risk regimes: {current_signal}"
    },
    "annotate": {
        "notes": "Values indexed to 100 for comparison"
    },
    "publish": {
        "embed-width": 700,
        # "chart-height": 800,
    }
}
# Update and publish
dw.update_chart(chart['id'], metadata=metadata)
dw.publish_chart(chart['id'])


published_url_sandp= dw.get_chart_display_urls(chart['id']); published_url_sandp

# In[59]:


df_plot = df_risks.tail(720).copy()
# df_plot.index = df_risks.index
df_plot = df_plot.reset_index()
df_plot['Date'] = df_plot['Date'].dt.strftime('%Y-%m-%d')


# In[60]:


# Calculate mean for each column (excluding Date)
means = df_plot.select_dtypes(include=['float64', 'int64']).mean()

# Create a copy of the original dataframe
df_plot = df_plot.copy()

# Subtract mean from each numeric column
for column in df_plot.select_dtypes(include=['float64', 'int64']).columns:
    df_plot[column] = df_plot[column] - means[column]


# Now df_plot_normalized contains the mean-centered values
# Each value represents the deviation from its column's average

# In[61]:


df_plot.head()

# In[68]:


from datawrapper import Datawrapper

# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Risk Components Over Time",
    chart_type="d3-area"
)



# Add the data
dw.add_data(chart['id'], data=df_plot)

# Configure visualization properties
# Configure visualization properties
properties = {
    "visualize": {
        "stack-to-100": False,
        "label-colors": False,
        "interpolation": "linear",
        "custom-colors": {
            "MARKET_RISK": "#15607a",
            "BUSINESS_RISK": "#1d81a2", 
            "POLITICAL_RISK": "#7eacbc",
            "TURING_RISK": "#ff4444"
        },
        "area-opacity": "0.8",
        "x-grid": "on",
        "y-grid": "on",
        "x-axis": {
            "tick-format": "%Y-%m-%d"
        },
        "legend": {
            "enabled": False
        },
        "show-color-key": False,
        "stack-color-legend": False,
        "tooltip": {
            "enabled": True,
            "show-title": True,
            "format": "0.00"
        },
        "show-tooltips": True,
        "plotHeightMode": "fixed",
        "plotHeightFixed": 300,
        "y-grid-subdivide": True,
        "y-grid-label-align": "left"
    },
    "describe": {
        "intro": "Stacked area chart showing the composition of Turing Risk over time, broken down into its main components: Market, Business, and Political risks.",
        "byline": "",
        "source-name": "Risk Analysis",
        "hide-title": False
    },
    "publish": {
        "embed-width": 900,
        "embed-height": 457,
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

# Update the chart with the properties
dw.update_chart(
    chart['id'],
    metadata=properties
)

# Publish the chart
dw.publish_chart(chart['id'])

# Get the published URL
published_url_area = dw.get_chart_display_urls(chart['id'])
print("Published Chart URL:", published_url)

# In[26]:


# Create lagged version of Turing Risk (t-1)
df_lagged = df_two.copy()
df_lagged['TURING_RISK_LAG'] = df_lagged['TURING_RISK'].shift(1)

# Remove the first row which will have NaN due to lag
df_lagged = df_lagged.dropna()

# Calculate correlation between lagged Turing Risk and S&P 500
correlation = df_lagged['TURING_RISK_LAG'].corr(df_lagged['SP500'])

print(f"Correlation between lagged Turing Risk and S&P 500: {correlation:.3f}")

# In[30]:


from datawrapper import Datawrapper
# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")
# Create the chart
chart = dw.create_chart(
    title=f"Lagged Turing Risk vs S&P 500 (Correlation: {correlation:.3f})",
    chart_type="d3-scatter-plot"
)

# Add the data
dw.add_data(chart['id'], data=df_lagged[['TURING_RISK_LAG', 'SP500']])

# Configure visualization properties
properties = {
    "visualize": {
        "x-grid": True,
        "y-grid": True,
        "x-axis-label": "Lagged Turing Risk",
        "y-axis-label": "S&P 500",
        "scale-x": "linear",
        "scale-y": "linear",
        "custom-colors": {
            "points": "#18a1cd",
            "regression-line": "#333333"
        },
        "regression-type": "linear",
        "point-size": 8,  # Increased from 4 to 8
        "fixed-size": 15,  # Added for larger bubbles
        "opacity": 0.6,
        "show-regression-line": True,
        "label-x": "Lagged Turing Risk",
        "label-y": "S&P 500",
        "format-x": "0.00",
        "format-y": "0.00",
        "grid-lines": "show",
        "regression": True,  # Explicitly enable regression
        "regression-method": "linear",
        "plotHeightMode": "fixed",
        "plotHeightFixed": 400,  # Increased plot height
        "hover-highlight": True,
        "tooltip": {
            "enabled": True,
            "body": "",
            "title": "",
            "sticky": False
        }
    },
    "describe": {
        "intro": (
            "Scatter plot showing the predictive relationship between Turing Risk (lagged by 1 period) "
            f"and S&P 500 returns. The strong negative correlation ({correlation:.3f}) indicates that "
            "higher Turing Risk typically precedes lower S&P 500 values."
        ),
        "byline": "",
        "source-name": "Turing Risk Analysis",
        "hide-title": False
    },
    "publish": {
        "embed-width": 800,
        "embed-height": 600,
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

# In[37]:


# Define title
page_title = "Predict a Monthly Bird"

# Define content sections using the content_sections dictionary
content_sections  = {
    "section_1": {
        "heading": "Tracking Risks",
        "content": "The Turing Risk indicator combines multiple risk factors into a single dynamic measure that evolves daily to capture market conditions.",
        "url": published_url_lines[0]["url"],
        "list": None
    },
    "section_2": {
        "heading": "Component Area Risks",
        "content": "Looking at recent changes in across 60 different types of risks for the last year",
        "url": published_url_area[0]["url"],
        "list": None
    },
    "section_3": {
        "heading": "Business Risks",
        "content": "Business risk indicators track company-specific factors including operational efficiency, earnings stability, and competitive positioning.",
        "url": published_url_business[0]["url"],
        "list": None
    },
    "section_4": {
        "heading": "Market Risks",
        "content": "Market risk factors measure systematic exposures including volatility, liquidity conditions, and broad market sentiment.",
        "url": published_url_market[0]["url"],
        "list": None
    },
    "section_5": {
        "heading": "Short-Term Risk Management",
        "content": "The strong negative relationship between Turing Risk and S&P 500 demonstrates the indicator's predictive power for market movements.",
        "url": published_url_sandp[0]["url"],
        "list": None
    },
    "section_6": {
        "heading": "Correlation with S&P Returns",
        "content": "The -0.735 correlation coefficient between lagged Turing Risk and S&P 500 returns indicates significant predictive ability for market direction.",
        "url": published_url[0]["url"],
        "list": None
    }
}


# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

