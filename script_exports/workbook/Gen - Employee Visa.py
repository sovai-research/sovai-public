#!/usr/bin/env python
# coding: utf-8

# In[ ]:


!pip install uv && uv pip install sovai['full'] --system > output.log 2>&1

# In[4]:


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


# In[5]:


import sovai as sov
import pandas as pd

sov.token_auth(token="visit https://sov.ai/profile for your token")

tickers_meta = pd.read_parquet("data/tickers.parq")

df_visa = sov.data("visas/h1b", full_history=True)


# Attempting URL 2: https://nyc3.digitaloceanspaces.com/sovai/sovai-accounting/processed/ratios_percentile_weekly

def calculate_visa_stats(df_visa, period='year', max_date=None):
    """
    Calculate visa statistics for either a year or quarter period.
    
    Parameters:
    df_visa (DataFrame): Input DataFrame with visa data
    period (str): 'year' or 'quarter'
    max_date (Timestamp, optional): Maximum date to use. If None, uses max date in data
    
    Returns:
    DataFrame: Statistics by ticker
    """
    # Get max_date if not provided
    if max_date is None:
        max_date = df_visa.index.get_level_values('date').max()
    
    # Calculate start_date and number of months based on period
    if period == 'year':
        start_date = max_date - pd.DateOffset(years=1)
        months_in_period = 12
    elif period == 'quarter':
        start_date = max_date - pd.DateOffset(months=3)
        months_in_period = 3
    else:
        raise ValueError("Period must be either 'year' or 'quarter'")
    
    # Filter the data for the period
    df_period = df_visa[df_visa.index.get_level_values('date') >= start_date]

    df_period = df_period.filter(["market_cap>100"])
    
    # Create a mask for rows where all employment types are 0
    empty_employment_mask = (
        (df_period['new_employment'] == 0) & 
        (df_period['continued_employment'] == 0) & 
        (df_period['change_previous_employment'] == 0) & 
        (df_period['new_concurrent_employment'] == 0) & 
        (df_period['change_employer'] == 0)
    )
    
    # Set new_employment to 1 where the mask is True
    df_period.loc[empty_employment_mask, 'new_employment'] = 1
    
    # Group by ticker and calculate statistics
    ticker_stats = df_period.groupby('ticker').agg({
        'predicted_pay': 'median',
        'case_status': lambda x: (x != 'certified').mean(),
        'new_employment': 'sum',
        'continued_employment': 'sum',
        'total_worker_positions': 'sum'  # Count number of applications
    }).round(4)
    
    # Calculate applications per month
    ticker_stats['apps_pm'] = (ticker_stats['total_worker_positions'] / months_in_period).round(4)
    
    # Calculate combined change_employer metric
    change_employer_sum = df_period.groupby('ticker')[
        ['change_previous_employment', 'new_concurrent_employment', 'change_employer']
    ].sum().sum(axis=1)
    
    # Add to ticker_stats
    ticker_stats['change_employer'] = change_employer_sum
    
    # Calculate percentages based on total applications
    total_apps = ticker_stats['total_worker_positions']
    ticker_stats['pct_new_employment'] = (ticker_stats['new_employment'] / total_apps).round(4)
    ticker_stats['pct_continued_employment'] = (ticker_stats['continued_employment'] / total_apps).round(4)
    ticker_stats['pct_change_employer'] = (ticker_stats['change_employer'] / total_apps).round(4)
    
    # Drop the raw count columns
    ticker_stats = ticker_stats.drop(columns=['total_worker_positions', 'new_employment', 'continued_employment', 'change_employer'])
    
    # Final column names
    ticker_stats.columns = [
        'median_pay',
        'denial_rate',
        'apps_pm',
        'pct_new',
        'pct_cont',
        'pct_change'
    ]
    
    return ticker_stats

# Usage examples:
max_date = df_visa.index.get_level_values('date').max()
# Calculate annual statistics
annual_stats = calculate_visa_stats(df_visa, period='year', max_date=max_date)
# Calculate quarterly statistics
quarterly_stats = calculate_visa_stats(df_visa, period='quarter', max_date=max_date)

quarterly_stats = quarterly_stats[quarterly_stats["apps_pm"]>4]

quarterly_stats

# Select annual metrics we want to compare
annual_comparison = annual_stats[['median_pay', 'denial_rate', 'apps_pm']].rename(
    columns={
        'median_pay': 'annual_median_pay',
        'denial_rate': 'annual_denial_rate', 
        'apps_pm': 'annual_apps_pm'
    }
)

# Merge with quarterly stats
quarterly_stats = quarterly_stats.merge(
    annual_comparison,
    left_index=True,
    right_index=True,
    how='left'
)

# Rename columns to shorter versions
quarterly_stats.columns = [
    'med_pay_q',  # quarterly median_pay
    'deny_q',     # quarterly denial_rate
    'apps_q',     # quarterly apps_pm
    'pct_new',    # pct_new stays the same as it's already short
    'pct_cont',   # pct_cont stays the same
    'pct_chg',    # pct_change
    'med_pay_y',  # annual median_pay
    'deny_y',     # annual_denial_rate
    'apps_y'      # annual_apps_pm
]



# Previous merging code stays the same...

# Multiply percentage columns by 100 for quarterly stats
percentage_columns = ['deny_q', 'deny_y']
quarterly_stats[percentage_columns] = quarterly_stats[percentage_columns] * 100

# Calculate percentage point difference in denial rates (quarterly - yearly)
quarterly_stats['deny_diff'] = (quarterly_stats['deny_q'] - quarterly_stats['deny_y']).round(2)

# Calculate percentage change in applications
quarterly_stats['apps_chg'] = ((quarterly_stats['apps_q'] - quarterly_stats['apps_y'])/quarterly_stats['apps_y'] * 100).round(2)

# Calculate percentage change in median pay
quarterly_stats['pay_chg'] = ((quarterly_stats['med_pay_q'] - quarterly_stats['med_pay_y'])/quarterly_stats['med_pay_y'] * 100).round(2)

# Reorder and rename columns
quarterly_stats = quarterly_stats[[
    # Pay metrics
    'med_pay_q', 
    'med_pay_y', 
    'pay_chg',
    # Application metrics
    'apps_q', 
    'apps_y', 
    'apps_chg',
    # Denial metrics
    'deny_q', 
    'deny_y', 
    'deny_diff',
    # Employment type percentages
    'pct_new',
    'pct_cont',
    'pct_chg'
]].rename(columns={
    'med_pay_q': 'pay_q',
    'med_pay_y': 'pay_y',
    'pay_chg': 'pay_d',
    'apps_q': 'app_q',
    'apps_y': 'app_y',
    'apps_chg': 'app_d',
    'deny_q': 'den_q',
    'deny_y': 'den_y',
    'deny_diff': 'den_d',
    'pct_new': 'new',
    'pct_cont': 'cont',
    'pct_chg': 'chg'
})

# Prepare the data
quarterly_stats = quarterly_stats.reset_index()  # Reset index to get ticker as column

# Convert pay to thousands (K)
quarterly_stats['pay_q'] = quarterly_stats['pay_q'] / 1000
quarterly_stats['pay_y'] = quarterly_stats['pay_y'] / 1000


# Previous merging code stays the same...

# Round all numeric columns to integers
numeric_columns = [
    'new', 'cont', 'chg'
]

# Round all numeric columns to integers (for display)
for col in numeric_columns:
    quarterly_stats[col] = (quarterly_stats[col] *100).round(2)



# Round all numeric columns to integers
numeric_columns = [
    'pay_q', 'pay_y', 'pay_d', 'app_q', 'app_y', 'app_d','new', 'cont', 'chg','den_q','den_y','den_d'
]

# Round all numeric columns to integers (for display)
for col in numeric_columns:
    quarterly_stats[col] = quarterly_stats[col].round(0).astype(int)

quarterly_stats.sort_values("new")

# In[6]:


quarterly_stats_small = quarterly_stats.drop(columns=["app_y","den_y"])

# In[7]:


from datawrapper import Datawrapper
import pandas as pd
# Initialize Datawrapper
dw = Datawrapper(access_token="your_token")

# Create the chart
chart = dw.create_chart(
    title="Foreign Employment Analysis - H1B Applications",
    chart_type="tables"
)

# Add the data to the chart
dw.add_data(chart['id'], data=quarterly_stats_small)

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
            "pay_q": {
                "title": "Pay Q",
                "format": "$0K",
                "width": "120"
            },
            "pay_y": {
                "title": "Pay Y",
                "format": "$0K",
                "width": "120"
            },
            "pay_d": {
                "title": "Pay Δ",
                "format": "+0%",
                "width": 0.27,
                "showAsBar": True,
                "barColorNegative": "#ff4444",
                "fixedWidth": True
            },
            "app_q": {
                "title": "Apps Q",
                "format": "0",
                "width": "100"
            },
            "app_y": {
                "title": "Apps Y",
                "format": "0",
                "width": "100"
            },
            "app_d": {
                "title": "Apps Δ",
                "format": "+0%",
                "width": 0.27,
                "showAsBar": True,
                "barColorNegative": "#ff4444",
                "fixedWidth": True
            },
            "den_q": {
                "title": "Deny Q",
                "format": "0%",
                "width": "100"
            },
            "den_y": {
                "title": "Deny Y",
                "format": "0%",
                "width": "100"
            },
            "den_d": {
                "title": "Deny Δ",
                "format": "+0%",
                "width": 0.27,
                "showAsBar": True,
                "barColorNegative": "#ff4444",
                "fixedWidth": True
            },
            "new": {
                "title": "New",
                "format": "0%",
                "width": "80"
            },
            "cont": {
                "title": "Cont",
                "format": "0%",
                "width": "80"
            },
            "chg": {
                "title": "Chg",
                "format": "0%",
                "width": "80"
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
        "firstColumnIsSticky": True,
        "mergeEmptyCells": False
    }
}

properties["describe"] = {
    "intro": "Analysis of H1B visa applications showing quarterly (Q) vs yearly (Y) comparisons. Pay in thousands (K), denial rates and employment types in percentages.",
    "byline": "Pay Q/Y: Median salary quarterly/yearly | " +
              "Pay Δ: Salary percentage change | " +
              "Apps Q/Y: Applications per month | " +
              "Apps Δ: Application volume change | " +
              "Deny Q/Y: Application rejection rate | " +
              "Deny Δ: Rejection rate change | " +
              "New: New employment share | " +
              "Cont: Continued employment share | " +
              "Chg: Employment change share",
    "source-name": "H1B Visa Data",
    "source-url": "",
    "hide-title": False
}


properties["publish"] = {
    "embed-width": 1200,
    "embed-height": 886,
    "blocks": {
        "logo": {"enabled": False},
        "embed": False,
        "download-pdf": False,
        "download-svg": False,
        "get-the-data": True,
        "download-image": False
    },
    "autoDarkMode": False,
    "chart-height": 788,
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

# In[8]:


# Define title
page_title = "Predict a Monthly Bird"

# Define content sections using the content_sections dictionary
content_sections  = {
    "section_1": {
        "heading": "Employee Visa Applications",
        "content": "You can see how many employees are poached from other companies as well as how many new foreign hires you employed",
        "url": published_url[0]["url"],
        "list": None
    },


}

# Handle page creation or append
handle_page_creation_or_append(page_title, DATABASE_ID, content_sections)
