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


# In[4]:


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


# In[5]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")


def get_bankruptcy_date(frequency="difference"):

    df = sov.data("bankruptcy", frequency=frequency)
    
    df = df.set_index(["ticker","date"])
    
    df = df.filter(["market_cap>100"])
    
    df[["sans_market","probability"]].reset_index()
    
    df = pd.merge(df[["sans_market","probability"]].reset_index(),tickers_meta[["ticker","sector"]], on="ticker", how="left")
    
    df = df.rename(columns={"sans_market":"sansmarket"})
    return df


# In[12]:


import pandas as pd
import io

def create_sector_ranking(df, sort_column='sansmarket', top_n=10, select='both'):
    def make_clickable(ticker):
        url = f"https://finance.yahoo.com/quote/{ticker}"
        return f'<a href="{url}" target="_blank">{ticker}</a>'
    
    # Ensure we're working with the latest date
    latest_date = df['date'].max()
    df_latest = df[df['date'] == latest_date]
    
    # Group by sector and get top N largest and smallest for each sector
    grouped = df_latest.groupby('sector')
    top_n_df = pd.DataFrame()
    
    for name, group in grouped:
        if select in ['largest', 'both']:
            top_largest = group.nlargest(top_n, sort_column)
            top_largest = top_largest.copy()
            top_largest['rank_type'] = 'Largest'
            top_largest['rank'] = range(1, len(top_largest) + 1)
            top_n_df = pd.concat([top_n_df, top_largest])
        
        if select in ['smallest', 'both']:
            top_smallest = group.nsmallest(top_n, sort_column)
            top_smallest = top_smallest.copy()
            top_smallest['rank_type'] = 'Smallest'
            top_smallest['rank'] = range(1, len(top_smallest) + 1)
            top_n_df = pd.concat([top_n_df, top_smallest])
    
    # Format the sort_column to two decimal places
    top_n_df[sort_column] = top_n_df[sort_column].map("{:.2f}".format)
    
    # Make tickers clickable
    top_n_df['ticker'] = top_n_df['ticker'].apply(make_clickable)
    
    # Sort the DataFrame for better organization
    top_n_df.sort_values(['sector', 'rank_type', 'rank'], inplace=True)
    
    # Create a pivot table with the rank_type and rank as multi-index
    top_n_pivot = top_n_df.pivot_table(index=['rank_type', 'rank'], 
                                      columns='sector', 
                                      values=[sort_column, 'ticker'],
                                      aggfunc='first')
    top_n_pivot = top_n_pivot.swaplevel(axis=1).sort_index(axis=1, level=0)
    
    # Prepare the data for Datawrapper format
    output = io.StringIO()
    
    # Write the first row (sector names)
    sectors = top_n_pivot.columns.get_level_values(0).unique()
    output.write('Type,Rank,')
    for sector in sectors:
        output.write(f'~~~{sector}~~~,,')
    output.write('\n')
    
    # Write the second row (sans_market and ticker)
    output.write(', ,')
    for _ in sectors:
        output.write(f'{sort_column},Ticker,')
    output.write('\n')
    
    # Write the data rows
    for (rank_type, rank), row in top_n_pivot.iterrows():
        output.write(f'{rank_type},{rank},')
        for sector in sectors:
            sans_market = row.get((sector, sort_column), '')
            ticker = row.get((sector, 'ticker'), '')
            output.write(f'{sans_market},{ticker},')
        output.write('\n')
    
    # Get the CSV string
    csv_string = output.getvalue()
    output.close()
    
    return csv_string



from datetime import datetime
import pandas as pd
# Unified configuration dictionary
MODEL_CONFIG = {
    'sansmarket': {
        'model_name': "Sans Market Model",
        'intro': (
            "The Sans Market model has fundamental accounting values pointing to bankruptcy, "
            "but not the market price and behaviour. It is a leading indicator with good short potential | "
            '<a href="https://sov.ai">Sov.ai™</a>'
        ),
        'start_title': {
            True: "Change in Bankruptcy Prediction",
            False: "Level of Bankruptcy Prediction"
        },
        'data_function': get_bankruptcy_date,
        'title_template': "{start_title} - {model_name} ({month_year})"
    },
    'probability': {
        'model_name': "Ensemble Model",
        'intro': (
            "The Ensemble Model uses deep learning and other models and multiple datasets to predict bankruptcy, "
            "it is a concurrent indicator with reasonable short potential | "
            '<a href="https://sov.ai">Sov.ai™</a>'
        ),
        'start_title': {
            True: "Change in Bankruptcy Prediction",
            False: "Level of Bankruptcy Prediction"
        },
        'data_function': get_bankruptcy_date,
        'title_template': "{start_title} - {model_name} ({month_year})"
    },

}

def generate_title_and_intro(sort_column, frequency):
    """
    Generates the title, retrieves the intro, and returns the data function based on sort_column and frequency.
    
    Parameters:
    - sort_column (str): The column to sort by.
    - frequency (bool): Determines the start title based on change or level.
    
    Returns:
    - tuple: (title, intro, data_function)
    """
    if sort_column not in MODEL_CONFIG:
        valid_columns = ', '.join(MODEL_CONFIG.keys())
        raise ValueError(f"Invalid sort_column: '{sort_column}'. Available options are: {valid_columns}")
    
    config = MODEL_CONFIG[sort_column]
    
    # Convert frequency to boolean; treat None as False
    frequency_bool = bool(frequency)
    
    # Get the appropriate start_title based on frequency
    start_title = config['start_title'].get(frequency_bool, f"{config['model_name']} Prediction")
    
    # Get current date in "Month Year" format
    current_date = datetime.now()
    month_year = current_date.strftime("%B %Y")  # e.g., "October 2024"
    
    # Construct the full title using the template
    title = config['title_template'].format(
        start_title=start_title,
        model_name=config['model_name'],
        month_year=month_year
    )
    
    # Get the intro text
    intro = config['intro']
    
    # Get the data function
    data_function = config['data_function']
    
    return title, intro, data_function



from datawrapper import Datawrapper
from IPython.display import IFrame
from datetime import datetime

def get_or_create_chart(dw, title, chart_type):
    # Try to find an existing chart with the given title
    charts = dw.get_charts(search=title, limit=1)

    charts['total'] = 0 ##  Force to remove update function
    
    if charts['total'] > 0:
        # If a chart is found, return its ID
        print("Existing chart found:")
        return charts['list'][0]['id']
    else:
        # If no chart is found, create a new one
        print("No existing chart found. Creating a new one.")
        new_chart = dw.create_chart(title=title, chart_type=chart_type)
        print("New chart created:")
        return new_chart['id']



def get_color_scheme(selection):
    if selection == "largest":
        return [
            {"color": "#f0f9e8", "position": 0},
            {"color": "#b6e3bb", "position": 0.16666666666666666},
            {"color": "#75c8c5", "position": 0.3333333333333333},
            {"color": "#4ba8c9", "position": 0.5},
            {"color": "#2989bd", "position": 0.6666666666666666},
            {"color": "#0a6aad", "position": 0.8333333333333334},
            {"color": "#254b8c", "position": 1}
        ]
    elif selection == "smallest":
        return [
            {"color": "#fff5f0", "position": 0},
            {"color": "#fee0d2", "position": 0.16666666666666666},
            {"color": "#fcbba1", "position": 0.3333333333333333},
            {"color": "#fc9272", "position": 0.5},
            {"color": "#fb6a4a", "position": 0.6666666666666666},
            {"color": "#ef3b2c", "position": 0.8333333333333334},
            {"color": "#cb181d", "position": 1}
        ]
    elif selection == "both":
        return [
            {"color": "#f0f9e8", "position": 0},
            {"color": "#b6e3bb", "position": 0.16666666666666666},
            {"color": "#75c8c5", "position": 0.3333333333333333},
            {"color": "#fc9272", "position": 0.5},
            {"color": "#fb6a4a", "position": 0.6666666666666666},
            {"color": "#ef3b2c", "position": 0.8333333333333334},
            {"color": "#cb181d", "position": 1}
        ]
    else:
        raise ValueError("selection must be either 'largest' or 'smallest'")

# Usage:




def create_sort_table(sort_column='slopebreak', frequency=None, top_n=10, selection="largest"):
    """
    Creates and updates a Datawrapper table with appropriate heatmap settings.
    
    Parameters:
    - sort_column (str): The column to sort by ('slopebreak' or 'predictbreak').
    - frequency (bool or None): Determines if it's a change or level metric.
    - top_n (int): Number of top entries per sector.
    - selection (str): 'largest', 'smallest', or 'both'.
    
    Returns:
    - tuple: (public_url, IFrame object)
    """
    # Generate title, intro, and data function based on sort_column and frequency
    title, intro, data_func = generate_title_and_intro(sort_column, frequency)

    # Fetch and process the data
    df = data_func(frequency=frequency)
    csv_data = create_sector_ranking(df, sort_column=sort_column, top_n=top_n, select=selection)

    # Initialize Datawrapper with your access token
    dw = Datawrapper(access_token="your_token")

    # Create or get existing chart
    chart_id = get_or_create_chart(dw, title, "tables")
    print(f"Chart ID: {chart_id}")

    try:
        # Update the chart with new data
        dw.add_data(chart_id, data=csv_data)
        print("Data added successfully")
    except Exception as e:
        print(f"Error adding data: {str(e)}")
        return None

    # Define sectors based on your data
    sectors = ['Energy', 'Utilities', 'Healthcare', 'Technology', 'Industrials', 
               'Real Estate', 'Basic Materials', 'Consumer Cyclical', 
               'Consumer Defensive', 'Financial Services', 'Communication Services']

    # Dynamically generate heatmap color palette
    # Adjust the colors and positions as per your requirements
    heatmap_colors = [
        {"color": "#2166ac", "position": 0},
        {"color": "#67a9cf", "position": 0.16666666666666666},
        {"color": "#d1e5f0", "position": 0.3333333333333333},
        {"color": "#f8f6e9", "position": 0.5},
        {"color": "#fddbc7", "position": 0.6666666666666666},
        {"color": "#ef8a62", "position": 0.8333333333333334},
        {"color": "#b2182b", "position": 1} 
    ]

    # Determine appropriate heatmap ranges based on data
    sort_min = df[sort_column].min()
    sort_max = df[sort_column].max()

    # Prepare the 'columns' configuration
    columns_config = {
        "": {"align": "left", "title": "Rank", "width": "auto"},
        "Type": {"align": "left", "title": "Type", "width": "auto"},
    }

    for sector in sectors:
        columns_config[f"~~~{sector}~~~"] = {
            "format": "0.[00]%",  # Adjust format as needed
            "heatmap": {
                "enabled": True
            }
        }

    # Define the updated_metadata with detailed heatmap settings
    updated_metadata = {
        "data": {
            "transpose": False,
            "vertical-header": True,
            "horizontal-header": True,
            "upload-method": "copy"
        },
        "visualize": {
            "dark-mode-invert": True,
            "perPage": 10,  # Items per page
            "pagination": {
                "enabled": True,
                "position": "bottom",
                "pagesPerScreen": 10
            },
            "highlighted-series": [],
            "highlighted-values": [],
            "sharing": {
                "enabled": False,
                "url": f"https://www.datawrapper.de/_/{chart_id}",
                "auto": False
            },
            "rows": {
                "header": {"rows": 2},
                "row--1": {
                    "style": {
                        "bold": False,
                        "color": False,
                        "italic": False,
                        "fontSize": 1,
                        "underline": False,
                        "background": False
                    },
                    "format": "0,0.[00]",
                    "moveTo": "top",
                    "sticky": False,
                    "moveRow": False,
                    "stickTo": "top",
                    "borderTop": "none",
                    "borderBottom": "none",
                    "borderTopColor": "#333333",
                    "overrideFormat": False,
                    "borderBottomColor": "#333333"
                }
            },
            "header": {
                "style": {
                    "bold": True,
                    "color": False,
                    "italic": False,
                    "fontSize": 1.1,
                    "background": False
                },
                "borderTop": "none",
                "borderBottom": "2px",
                "borderTopColor": "#333333",
                "borderBottomColor": "#333333"
            },
            "legend": {
                "size": 170,
                "labels": "ranges",
                "enabled": False,
                "reverse": False,
                "labelMax": "high",
                "labelMin": "low",
                "position": "above",
                "interactive": False,
                "labelCenter": "medium",
                "labelFormat": "0,0.[00]",
                "customLabels": []
            },
            "columns": columns_config,
            "heatmap": {
                "map": {},
                "mode": "continuous",
                "stops": "equidistant",
                "palette": 0,
                "rangeMax": "",  # Let Datawrapper auto-calculate or set manually
                "rangeMin": "",
                "stopCount": 5,
                "hideValues": False,
                "customStops": [],
                "rangeCenter": "30",  # Adjust based on your data
                "categoryOrder": [],
                "interpolation": "deciles",
                "categoryLabels": {},
                "colors": heatmap_colors  # Custom color palette
            },
            "perPage": 10,
            "striped": False,
            "markdown": True,
            "showRank": False,
            "sortTable": False,
            "pagination": {"enabled": True, "position": "bottom"},
            "searchable": False,
            "showHeader": True,
            "compactMode": True,
            "sortDirection": "desc",
            "chart-type-set": True,
            "mobileFallback": False,
            "mergeEmptyCells": True,
            "firstRowIsHeader": True,
            "firstColumnIsSticky": True
        },
        "describe": {
            "intro": intro,
            "byline": "",
            "source-name": "",
            "source-url": "",
            "hide-title": False
        },
        "publish": {
            "embed-width": 600,
            "embed-height": 510,
            "blocks": {
                "logo": {"enabled": False},
                "embed": False,
                "download-pdf": False,
                "download-svg": False,
                "get-the-data": True,
                "download-image": False
            },
            "autoDarkMode": False,
            "chart-height": 395,
            "force-attribution": False
        }
    }

    # # Optionally, adjust 'rangeCenter' based on sort_column
    # if sort_column == 'slopebreak':
    updated_metadata['visualize']['heatmap']['rangeCenter'] = "0"  # Example adjustment
    # elif sort_column == 'predictbreak':
    #     updated_metadata['visualize']['heatmap']['rangeCenter'] = "30"  # Example adjustment

    # Update the chart with the new metadata
    try:
        result = dw.update_chart(chart_id, metadata=updated_metadata)
        print("Chart updated successfully")
    except Exception as e:
        print(f"Error updating chart: {str(e)}")
        return None

    # Publish the updated chart
    try:
        publish_result = dw.publish_chart(chart_id)
        print("Chart published successfully")
    except Exception as e:
        print(f"Error publishing chart: {str(e)}")
        return None

    # Retrieve and print the published chart URL
    try:
        published_url = dw.get_chart_display_urls(chart_id)
        print("Published Chart URL:", published_url)
    except Exception as e:
        print(f"Error getting chart URL: {str(e)}")
        return None

    # Extract the public URL from the publish result
    public_url = publish_result["data"]["publicUrl"]

    # Display the chart within the notebook
    return public_url, IFrame(src=public_url, width=1200, height=600)


# Call the function
# url_levels, frame  = create_sort_table(sort_column='slopebreak', frequency=None, top_n=40, selection="both")
# Call the function
url_sansmarket, frame = create_sort_table(sort_column='sansmarket', frequency="difference", top_n=20, selection="both")

# In[13]:


url_probs, frame = create_sort_table(sort_column='probability', frequency="difference", top_n=20, selection="both")

# In[14]:


# Define title
page_title = "Predict a Monthly Bird"

# Define content sections using the content_sections dictionary
content_sections  = {
    "section_1": {
        "heading": "Changes in Sans-Market Bankruptcy Probability",
        "content": ("The sans-market model is interesting because it looks at the bankruptcy using only fundamental data and highlights potential disconnect with the"
        " stock price. Hence these companies fundamentals looks good/bad, but their market valuations are low/high"),
        "url": url_sansmarket,
        "list": None
    },
    "section_2": {
        "heading": "Changes in Bankruptcy Probability",
        "content": "This looks at changes in the bankruptcy probability using all underlying data, and is sensitive to recent stock price declines, unlike the sans-market model.",
        "url": url_probs,
        "list": None
    },

}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)

# In[ ]:


# !pip install chart_studio

# import chart_studio
# chart_studio.tools.set_credentials_file(username='snowde', api_key='AdSAPaBisfTTFZr6Szxw')

# # # Install necessary packages
# # !pip install PyGithub



# fig = sov.data("risks/plot/turing_risk_plot", plot=True)

# # Export the figure to an HTML string
# plot_html = fig.to_html(full_html=True)

# # Call the function to upload the file
# upload_file_to_github(plot_html, file_path_in_repo)



# import chart_studio.plotly as py
# py.iplot(fig, filename='turing_risk_plot')

# fig = sov.plot("bankruptcy", chart_type="shapley", tickers=["AMTX"])

# import chart_studio.plotly as py
# py.iplot(fig, filename='example_explain_bankruptcy_ticker')

# fig

# sov.plot("bankruptcy", chart_type="line", tickers=["AMTX"])

# dw.publish_chart("qv6a1")

# dw.publish_chart("bUu61")

# https://datawrapper.dwcdn.net/QabIN/1/
